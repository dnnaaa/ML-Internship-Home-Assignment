
from .eda_component import EDAComponent
from .training_component import TrainingComponent
from .inference_component import InferenceComponent


class ComponentManager:
    def __init__(self):
        self.components = {
            "EDA": EDAComponent(),
            "Training": TrainingComponent(),
            "Inference": InferenceComponent(),
            ##If you want to add some new component you add here , you just new to create a Newcomponent Class that extends from the base_component
        }

    def get_component(self, name):
        """
        Retrieve a component by name.
        """
        return self.components.get(name)
