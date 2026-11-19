N = 10

masses = [1] * N
stiffness = [1000] * N
damping = [2] * N

sensor_dofs = [0, 2, 4, 5, 7, 9]

damage_elements = list(range(N))
damage_severities = [0.4,0.3,0.0,0.0,0.3,0.1,0.1,0.0,0.2,0.1]

forces = [
    {
        "floor": 0,
        "amplitude": 10.0,
        "angular_frequency": 5.0
    },
    {
        "floor": 1,
        "amplitude": 8.0,
        "angular_frequency": 4.0
    },
    {
        "floor": 2,
        "amplitude": 12.0,
        "angular_frequency": 6.0
    },
    {
        "floor": 4,
        "amplitude": 7.0,
        "angular_frequency": 3.0
    },
    {
        "floor": 5,
        "amplitude": 9.0,
        "angular_frequency": 5.0
    },
    {
        "floor": 7,
        "amplitude": 6.0,
        "angular_frequency": 2.0
    },
    {
        "floor": 9,
        "amplitude": 11.0,
        "angular_frequency": 4.0
    }
]


