from enum import StrEnum


class StatesModel(StrEnum):
    PENDING = "Pendiente"
    PROCESSING = "Procesando"
    COMPLETED = "Completado"
    FAILED = "Fallido"
    CANCELLED = "Cancelado"
