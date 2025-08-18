# my_module/_types.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports only happen during static type checking
    import huge_module
    import another_huge_module
    import third_party_lib

    # Define your type aliases
    HugeModuleReturn = huge_module.SomeClass
    AnotherHugeModuleData = another_huge_module.SpecificDataType
    ThirdPartyConfig = third_party_lib.ConfigurationObject
    # Add more type aliases as needed...

else:
    # Provide fallback types for runtime if not type checking
    HugeModuleReturn = object
    AnotherHugeModuleData = object
    ThirdPartyConfig = object