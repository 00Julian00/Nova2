# Style guide and general layout of the Nova 2 project.

## Style guide

### File structure (for .py files)
```python
"""
Description: Brief description of the purpose of the file.
"""

# Imports grouped and ordered
import standard_libaries

import third_party_libaries

from .local_module import LocalClass

class ClassName:
    """
    If this is a base class, a docstring here explains the purpose of the class.
    If this is an implementation class, the docstring is in the __init__ method.
    """

    def __init__(self) -> None: # All classes must have an __init__ method, even if it is not used. It is the first method in the class.
        """
        A docstring in the __init__ method explains the purpose of the class, if it is an implementation class.
        """
        pass

    # Methods in a class must be ordered in a way that the importand modules come first, followed by smaller helper methods.
    # Important modules must be ordered by expected execution order. Helper methods must be ordered by importance (how often it is used).

    # Public methods, as well as complex internal methods require a docstring like this:
    # If a public method is very simple or self explanatory, it does not require a docstring.
    def important_method_1(self, parameter1: type) -> return_type:
        """
        Brief description of what the method does. If there are no arguments, this docstring does not require an 'Arguments' section. Same thing goes for 'Returns' if the method always returns 'None'.

        Arguments:
            parameter1 (type): Description of the parameter.

        Returns:
            return_type: Description of the return value.
        """

    def important_method_2(self) -> None:
        pass

    def _helper_method_1(self) -> None:
        pass

    def _helper_method_2(self) -> None:
        pass

```

### Coding conventions
Variables, methods, functions and file names are in snake_case. Class names are in CamelCase. Constants are in UPPER_CASE.\
All functions and methods in implementation classes require full type hinting. Methods in base classes do not require type hinting. 'self' or 'cls' parameters also do not require type hinting.\
The type hints need to cover all possible types. A '|' can be used to list multiple type. Any 'container' types like 'list' also need to specify what types they are storing 'list[type]'.\
The names of variables and methods meant to be only used inside the class need to start with an underscore.\
Blocks of code, as well as classes, methods and functions must be grouped and seperated via a new line. There can never be more then one new line between code.

### Comments
Comments should be used to explain certain approaches. They should always aim to improve readability and understandability.\
There needs to be a space between the '#' and the actual comment.

### Special comments
Due to a VSCode plugin called 'Better Comments', there exist several symbols that change the color of a comment. These should be used like this:\
- #* This comment describes temporary or debug code. It is displayed light-green.
- #! This comment describes a critical problem or a very important information. It is displayed red.
- #? This comment describes a confusing part of code, for example if a piece of code does not work for unknown reasons. It is displayed blue.
- #TODO: This comment describes something that needs to be changed or implemented later. It is displayed orange.
These symbols are written directly after the '#', followed by a space and then the actual comment.

### Error handling
When catching an exception that makes it impossible for the program to continue running, a new exception has to be raised with the appropriate type and an error message that details the source of the exception in the context of the project.

### Raising and exception
If something goes wrong that makes it impossible for the program to continue running, an exception must be raised with the appropriate type and an error message that details the source of the exception in the context of the project.

### Best practices
Keep the design modular with every module having a clearly defined and narrow purpose.\
The code should remain readable and simple. Always consider outlining code if it better fits somewhere else.

## Project structure
- `api`: Holds the logic that acts as an interface between the base logic and any external logic, like tools.
- `app`: Holds the base logic. It also includes the `inference_engines` folder which holds several scripts called 'inference engines' that all provide the same API for LLM inference but use different approaches.
- `data`: Holds files that store small amounts of data, like config files or context files.
- `db`: Holds the databases.
- `tools`: Holds external tools.