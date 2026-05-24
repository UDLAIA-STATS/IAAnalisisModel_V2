from enum import StrEnum


class DepthModelsTypes(StrEnum):
    DPT_Large = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MiDaS_small = "MiDaS_small"

    @staticmethod
    def to_dict():
        types_dict = {}
        for type in DepthModelsTypes:
            types_dict[type.value] = type
        return types_dict