import torch

class ModelTool():
    def __init__(self) -> None:
        pass

    def save_checkpoint(self, checkpoint: dict,filename:str):
        """
            every model usually to save the checkpoint.
            so i wirte this tool and function.

            args:
                filename:
                    the path to save
        """
        torch.save(checkpoint, filename)


        