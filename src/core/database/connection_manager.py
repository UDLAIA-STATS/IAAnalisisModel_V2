import traceback

import logfire
from sqlmodel import SQLModel, Session, QueuePool, create_engine

from config.routes import DATABASE_DIR
from config.configuration import settings


class ConnectionManager():

    def __init__(self):
        """
        Constructor de la clase ConnectionManager.

        Args:
            match_id (int): ID del partido para el que se crea la base de datos.

        """
        self._create_engine()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.dispose()
    
    def _create_engine(self):
        try:
            database_url = settings.DATABASE_URL

            if not database_url:
                logfire.error("DATABASE_URL not found in settings file. Using local PostgreSQL fallback.")
                raise ValueError("DATABASE_URL not found in settings file. Using local PostgreSQL fallback.")

            self.engine = create_engine(
                database_url,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=15,
                pool_recycle=120000,
                echo=False
            )
        except ConnectionError as e:
            logfire.error(f"Error creating engine: {traceback.format_exc()}")
            raise e
        except Exception as e:
            logfire.error(f"Error creating engine: {traceback.format_exc()}")
            raise e

    def create_session(self):
        """
        Crea una sesion de base de datos que se puede utilizar para interactuar
        con la base de datos. La sesion se cierra automaticamente cuando se
        sale del ambito de la sesion.

        Returns:
            Session: Sesion de base de datos.
        """
        return Session(self.engine, expire_on_commit=False, autoflush=False)
    
    def dispose(self):
        self.engine.dispose()

    def create_database(self, drop_existing: bool = False):
        """
        Crea una base de datos temporal para el partido especificado por match_id.
        
        Se crea un motor de base de datos con una sesion de base de datos
        que se puede utilizar para interactuar con la base de datos. La sesion
        se cierra automaticamente cuando se sale del ambito de la sesion.
        
        La base de datos se crea en la carpeta especificada por DATABASE_DIR
        y se llama "temp_db_<match_id>.sqlite".
        
        Args:
            match_id (int): ID del partido para el que se crea la base de datos.
        """
        if drop_existing:
            SQLModel.metadata.drop_all(bind=self.engine)

        SQLModel.metadata.create_all(bind=self.engine)


connection_manager = ConnectionManager()