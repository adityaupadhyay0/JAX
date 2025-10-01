import timeit
import numpy as np
from python import axe

def benchmark(stmt, setup, number=20):
    """Runs and times a statement."""
    timer = timeit.Timer(stmt=stmt, setup=setup)
    return timer.timeit(number=number) / number

def main():
    """Runs and prints the benchmarks."""
    shape_a = (512, 512)
    shape_b = (512, 512)

    setup_np = f"""
import numpy as np
shape_a = {shape_a}
shape_b = {shape_b}
a = np.random.rand(*shape_a).astype(np.float32)
b = np.random.rand(*shape_b).astype(np.float32)
"""

    setup_axe = f"""
from python import axe
import numpy as np
shape_a = {shape_a}
shape_b = {shape_b}
a_np = np.random.rand(*shape_a).astype(np.float32)
b_np = np.random.rand(*shape_b).astype(np.float32)
a = axe.array(a_np)
b = axe.array(b_np)
"""

    benchmarks = {
        "Matrix Multiplication": {
            "numpy": "a @ b",
            "axe": "a @ b"
        },
        "Element-wise Addition": {
            "numpy": "a + b",
            "axe": "a + b"
        }
    }

    print(f"{'Operation':<25} | {'Library':<10} | {'Time (s)':<15}")
    print("-" * 55)

    for op_name, stmts in benchmarks.items():
        np_time = benchmark(stmts["numpy"], setup_np)
        axe_time = benchmark(stmts["axe"], setup_axe)

        print(f"{op_name:<25} | {'NumPy':<10} | {np_time:<15.6f}")
        print(f"{op_name:<25} | {'axe':<10} | {axe_time:<15.6f}")
        print("-" * 55)

if __name__ == "__main__":
    main()