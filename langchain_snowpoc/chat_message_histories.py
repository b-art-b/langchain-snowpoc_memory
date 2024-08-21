"""Client for persisting chat message history in Snowflake."""

from __future__ import annotations
import json
import logging
import re
import uuid
from typing import List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from snowflake.connector.connection import SnowflakeConnection

logger = logging.getLogger(__name__)


def _create_table_and_index(table_name: str) -> List[str]:
    """Make a SQL query to create a table."""
    index_name = f"idx_{table_name}_session_id"
    statements = [
        f"""
        CREATE HYBRID TABLE IF NOT EXISTS {table_name} (
            id NUMBER PRIMARY KEY AUTOINCREMENT START 1 INCREMENT 1,
            session_id CHAR(36) NOT NULL,
            message VARIANT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP()
        );
        """,
        f"""
        CREATE OR REPLACE INDEX {index_name} ON {table_name}(session_id);
        """,
    ]
    return statements


def _get_messages_query(table_name: str) -> str:
    """Make a SQL query to get messages for a given session."""
    return f"""SELECT message
        FROM {table_name}
        WHERE session_id = '%(session_id)s'
        ORDER BY created_at ASC;
        """


def _delete_by_session_id_query(table_name: str) -> str:
    """Make a SQL query to delete messages for a given session."""
    return f"""DELETE FROM {table_name} WHERE session_id = '%(session_id)s';"""


def _delete_table_query(table_name: str) -> str:
    """Make a SQL query to delete a table."""
    return f"""DROP TABLE IF EXISTS {table_name};"""


def _insert_message_query(table_name: str) -> str:
    """Make a SQL query to insert a message."""
    return f"""
        INSERT INTO {table_name} (session_id, message)
        SELECT '%s' as session_id, PARSE_JSON($$%s$$) as message
        """


class SnowflakeChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        table_name: str,
        session_id: str,
        /,
        *,
        connection: SnowflakeConnection,
    ) -> None:
        """Client for persisting chat message history in Snowflake.

        The client can create table in the database and provides methods to
        add messages, get messages, and clear the chat message history.

        The schema has the following columns:

        - id: A serial primary key.
        - session_id: The session ID for the chat message history.
        - message: The JSON message content.
        - created_at: The timestamp of when the message was created.

        Messages are retrieved for a given session_id and are sorted by
        the created_at, and correspond
        to the order in which the messages were added to the history.

        A session_id can be used to separate different chat histories in the same table,
        the session_id should be provided when initializing the client.

        This chat history client takes in a psycopg connection object
        and uses it to interact with the database.

        This design allows to reuse the underlying connection object across
        multiple instantiations of this class, making instantiation fast.

        This chat history client is designed for prototyping applications that
        involve chat and are based on Snowflake.

        As your application grows, you will likely need to extend the schema to
        handle more complex queries. For example, a chat application
        may involve multiple tables like a user table, a table for storing
        chat sessions / conversations, and this table for storing chat messages
        for a given session. The application will require access to additional
        endpoints like deleting messages by user id, listing conversations by
        user id or ordering them based on last message time, etc.

        Feel free to adapt this implementation to suit your application's needs.

        Args:
            session_id: The session ID to use for the chat message history
            table_name: The name of the database table to use
            connection: An existing Snowflake connection instance

        Usage:
            - Use the create_tables method to set up the table schema in the database.
            - Initialize the class with the appropriate session ID, table name,
              and database connection.
            - Add messages to the database using add_messages.
            - Retrieve messages with get_messages.
            - Clear the session history with clear when needed.

        Example:

        .. code-block:: python

            import uuid

            from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
            from langchain_snowpoc.chat_message_histories import SnowflakeChatMessageHistory

            # Establish a connection to the database
            import snowflake.connector

            CONNECTION_NAME = "arctic_user" # set your connection string

            snowflake_connection = snowflake.connector.connect(
                connection_name=CONNECTION_NAME,
            )

            # Create the table schema (only needs to be done once)
            table_name = "chat_history"
            SnowflakeChatMessageHistory.create_tables(snowflake_connection, table_name)

            session_id = str(uuid.uuid4())

            # Initialize the chat history manager
            chat_history = SnowflakeChatMessageHistory(
                table_name,
                session_id,
                connection=snowflake_connection
            )

            # Add messages to the chat history
            chat_history.add_messages([
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ])

            print(chat_history.messages)

            # you can check the content of the table manually
            import json
            dt = (
                snowflake_connection
                .cursor()
                .execute(
                f"select message from {table_name} where session_id='{session_id}' order by created_at ASC"
                )
                .fetchall()
            )
            for row in dt:
                print(json.loads(row[0])['data']['content'])
        """

        self._connection = connection

        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError(
                f"Invalid session id. Session id must be a valid UUID. Got {session_id}"
            )

        self._session_id = session_id

        if not re.match(r"^\w+$", table_name):
            raise ValueError(
                "Invalid table name. Table name must contain only alphanumeric "
                "characters and underscores."
            )
        self._table_name = table_name

    @staticmethod
    def create_tables(
        connection: SnowflakeConnection,
        table_name: str,
        /,
    ) -> None:
        """Create the table schema in the database and create relevant indexes."""
        queries = _create_table_and_index(table_name)
        logger.info("Creating table %s", table_name)
        with connection.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
        connection.commit()

    @staticmethod
    def drop_table(connection: SnowflakeConnection, table_name: str, /) -> None:
        """Delete the table schema in the database.

        WARNING:
            This will delete the given table from the database including
            all the database in the table and the schema of the table.

        Args:
            connection: The database connection.
            table_name: The name of the table to create.
        """

        query = _delete_table_query(table_name)
        logger.info("Dropping table %s", table_name)
        with connection.cursor() as cursor:
            cursor.execute(query)
        connection.commit()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the chat message history."""

        values = [
            (
                self._session_id,
                json.dumps(message_to_dict(message)),
            )
            for message in messages
        ]

        query = _insert_message_query(self._table_name)

        with self._connection.cursor() as cursor:
            for value in values:
                _q = query % (
                    value[0],
                    value[1],
                )
                cursor.execute(_q)
        self._connection.commit()

    def get_messages(self) -> List[BaseMessage]:
        """Retrieve messages from the chat message history."""
        query = _get_messages_query(self._table_name)

        with self._connection.cursor() as cursor:
            cursor.execute(query % {"session_id": self._session_id})
            items = [json.loads(record[0]) for record in cursor.fetchall()]

        messages = messages_from_dict(items)
        return messages

    @property
    def messages(self) -> List[BaseMessage]:
        """The abstraction required a property."""
        return self.get_messages()

    def clear(self) -> None:
        """Clear the chat message history for the GIVEN session."""
        if self._connection is None:
            raise ValueError(
                "Please initialize the SnowflakeChatMessageHistory "
                "with a sync connection or use the async clear method instead."
            )

        query = _delete_by_session_id_query(self._table_name)
        with self._connection.cursor() as cursor:
            cursor.execute(query, {"session_id": self._session_id})
        self._connection.commit()
