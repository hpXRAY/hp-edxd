import numpy as np
#import matplotlib.pyplot as plt

def generate_array_with_index(first_value_A, last_value_A, last_value_B, last_value_C, last_value_D, last_value_E, step_A, step_C, step_D, step_E):
    step_B = (step_A + step_C) / 2
    step_E = (step_D + step_E) / 2

    # Generate array A with the specified first and last values and step size
    array_A = np.arange(first_value_A, last_value_A, step_A)

    # Generate array B with the calculated average step size, matching the last value of array A
    array_B = np.arange(array_A[-1], last_value_B, step_B)

    # Generate array C with the specified last value and step size
    array_C = np.arange(array_B[-1], last_value_C, step_C)

    # Generate array D with the specified last value and step size
    array_D = np.arange(array_C[-1], last_value_D, step_D)

    # Generate array E with the calculated average step size, matching the last value of array D
    array_E = np.arange(array_D[-1], last_value_E, step_E)

    array_A = array_A[:-1]
    array_B = array_B[:-1]
    array_D = array_D[:-1]
    array_E = array_E[:-1]
    
    # Concatenate arrays A, B, C, D, and E
    result_array = np.concatenate((array_A, array_B, array_C, array_D, array_E))
    result_index = np.arange(len(result_array))

    return result_array, result_index

# Test the function and plot the result
first_value_A = 0.0
last_value_A = 10.0
last_value_B = 20.0
last_value_C = 30.0
last_value_D = 40.0
last_value_E = 50.0
step_A = 0.1
step_C = 0.2
step_D = 0.5
step_E = 0.6

result_array, result_index = generate_array_with_index(first_value_A, last_value_A, last_value_B, last_value_C, last_value_D, last_value_E, step_A, step_C, step_D, step_E)

'''# Plot x vs index
plt.plot(result_index, result_array, '-o')
plt.xlabel('Index')
plt.ylabel('x')
plt.title('Array Plot')
plt.grid(True)
plt.show()'''
