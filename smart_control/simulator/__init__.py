"""Building and HVAC simulation components.

This package provides a simulation environment for modeling the thermal
dynamics of buildings and the behavior of their Heating, Ventilation, and
Air Conditioning (HVAC) systems. It is designed to be used with reinforcement
learning agents for developing smart building control strategies.

Key components include:

- **Building Model**: Simulates the thermal properties of building zones,
  considering factors like construction materials, solar gain, and heat
  transfer between zones and with the outside environment.
- **HVAC System**: Models various HVAC components, including:
    - Air Handlers (AHUs): Manage air circulation, heating, and cooling.
    - Boilers: Provide hot water for heating.
    - Variable Air Volume (VAV) units: Control airflow to individual zones.
    - Thermostats: Sense zone temperatures and trigger heating/cooling actions.
- **Occupancy Models**: Simulate the presence and movement of occupants within
  the building, which impacts thermal loads and comfort requirements.
- **Weather Controller**: Provides realistic weather data (temperature, solar
  radiation) as input to the simulation.
- **Simulator Core**: Manages the overall simulation loop, time progression,
  and interactions between different components.
- **Smart Devices**: Abstract representations of controllable and observable
  devices within the simulation, conforming to a common interface.

The simulator allows for detailed modeling of building behavior and provides
an environment for training and evaluating RL agents aimed at optimizing
energy consumption and occupant comfort.
"""
