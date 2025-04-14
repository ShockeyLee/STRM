def fibonacci_sum(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        total = a + b
        for _ in range(2, n):
            a, b = b, a + b
            total += b
        return total

# Example usage
if __name__ == "__main__":
    n = 10  # Change this value to compute a different number of terms
    print(f"Sum of first {n} Fibonacci numbers: {fibonacci_sum(n)}")
