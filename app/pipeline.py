#!This script is in the experiemntal stage is not used currently
"""
Description: This script controls the execution flow of all the parts of the Nova system.
Data storage is handled by "context data". With these, any data that is needed can be saved and loaded at any time in the pipeline.
"""

class PipelineManager:
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def handle_block_execution_error(exception: Exception) -> None:
        pass

#Base processing block:
class ProcessingBlock:
    def __init__(
                self,
                name: str,
                description: str,
                base_logic: callable
                ) -> None:
        
        """
        Defines a processing block that can be added to the processing pipeline.
        If is_start_block, the block will be treated as an entry point and can not receive input data.
        base_logic is the callable that will be run when the block is activated.
        """

        self.name = name
        self.description = description

        self.base_logic = base_logic

class ProcessingBlockWrapper:
    def __init__(
                self,
                own_block: ProcessingBlock,
                next_block: ProcessingBlock
                ) -> None:
        """
        Used to control the execution flow.
        """

        self.own_block = own_block
        self.next_block = next_block

    def execute_block(self):
        try:
            self.own_block.base_logic()
        except Exception as exception:
            PipelineManager.handle_block_execution_error(exception=exception)

        self.next_block.execute_block()

#Specialized processing blocks:
class ForkProcess:
    def __init__(self) -> None:
        """
        Splits the execution flow of the pipeline using threading.
        """
        pass
    
    def execute_block(self):
        pass