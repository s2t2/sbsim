# Google Smart Buildings Control

This repository accompanies Goldfeder, J., Sipple, J., Real-World Data and
Calibrated Simulation Suite for Offline Training of Reinforcement Learning
Agents to Optimize Energy and Emission in Office Buildings, currently under
review at Neurips 2024, and builds off of Goldfeder, J., Sipple, J., (2023).
[A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings](https://dl.acm.org/doi/10.1145/3600100.3625682),
BuildSys '23, November 15–16, 2023, Istanbul, Turkey

## Real World Data

In addition to our calibrated simulator, we have released six years of data from
three buildings. This data can be used for further simulator calibration, and
for training and evaluating reinforcement learning (RL) models.

The dataset is available for download from
[Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/smart_buildings).

Alternatively, a smaller version of the dataset can be downloaded as a
[zip file](https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip)
from cloud storage.

## Documentation

View the official [Documentation Site](https://google.github.io/sbsim/) for a
complete auto-generated API reference.

There is also a legacy unofficial
[Community-run Documentation Site](https://gitwyd.github.io/sbsim_documentation/)
containing more information about the project and the codebase. We plan to merge
all this content into the official documentation site soon.

## Getting Started

A great place to start is by reviewing the
[Soft Actor Critic Demo notebook](smart_control/notebooks/SAC_Demo.ipynb). This
notebook will walk you through:

1. Creating a gym-compatible Reinforcement Learning (RL) environment.

2. Visualizing the environment.

3. Training an agent using the
   [Tensorflow Agents Library](https://www.tensorflow.org/agents).

Alternatively, RL agents can be trained by running various scripts in the
"smart_control/reinforcement_learning/scripts" directory.

Before running notebooks or scripts, make sure to complete the setup
instructions linked below.

## Setup

The [Setup Guide](docs/setup.md) provides all the information you need to run
the code locally.

## Contributing

The [Contributor's Guide](docs/contributing.md) provides more information on how
to contribute to this repository.

## Testing with Factories

This project uses [`factory_boy`](https://factoryboy.readthedocs.io/) to simplify the creation of test data and objects. Factories are defined in `smart_control/utils/factory_utils.py`.

### Using Factories

To use a factory, import it and then call it like a function. You can override default values by passing arguments:

```python
from smart_control.utils import factory_utils
from smart_control.proto import smart_control_building_pb2 # For type hinting or specific enums
from smart_control.utils import conversion_utils # For specific timestamp conversions if needed
import pandas as pd

# Example: Creating an ActionRequest
sar1 = factory_utils.SingleActionRequestFactory(device_id="my_device", continuous_value=50.0)
action_request = factory_utils.ActionRequestFactory(
    timestamp=conversion_utils.pandas_to_proto_timestamp(pd.Timestamp("2024-07-15 10:00:00")),
    single_action_requests=[sar1]
)

# Example: Creating a SimpleBuilding with custom start time
custom_start = pd.Timestamp("2023-01-01", tz="UTC")
building_instance = factory_utils.SimpleBuildingFactory(start_time=custom_start)
```

### Creating New Factories

1.  **Define the Factory**: Add a new class in `smart_control/utils/factory_utils.py` inheriting from `factory.Factory`.
2.  **Specify the Model**: In the `Meta` class, set `model` to the class or protobuf message you're creating.
    ```python
    class MyObjectFactory(factory.Factory):
        class Meta:
            model = MyActualClass # or path.to.MyProtoMessage
    ```
3.  **Define Attributes**:
    *   For simple values, assign them directly: `my_field = "default_value"`
    *   For dynamic values, use `factory.LazyFunction(lambda: some_function())` or `factory.Sequence(lambda n: f"item_{n}")`.
    *   For Faker data, use `factory.Faker('provider_name')`.
    *   For sub-factories (nested objects), use `factory.SubFactory(OtherFactory)`.
    *   For lists of sub-factories, use `factory.List([factory.SubFactory(OtherFactory) for _ in range(2)])` or pass them in during instantiation with a `factory.List([])` field and a `@factory.post_generation` hook if the list items need to be appended to a protobuf repeated field.

Refer to the `factory_boy` documentation for more advanced features like `post_generation` hooks, `Trait`s, etc.
The existing factories in `smart_control/utils/factory_utils.py` provide practical examples within this codebase.
For protobuf repeated fields that take a list of messages, define the field as `factory.List([])` and use a `@factory.post_generation` hook to extend the protobuf field with the items passed to the factory. Example: `ActionRequestFactory` for `single_action_requests`.

## [License](LICENSE)
