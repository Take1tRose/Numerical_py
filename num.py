import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from sympy import symbols, lambdify, simplify, cos, sin, expand, latex
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def gaussian_elimination(A, b):
    if np.linalg.det(A) == 0:
        return None

    augmented_matrix = np.column_stack((A, b))
    n = len(b)

    for i in range(n):
        diag = augmented_matrix[i, i]
        augmented_matrix[i, :] /= diag

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j, :] -= factor * augmented_matrix[i, :]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])
    return x

def newton_method(f, f_prime, initial_guess=None, tolerance=0.0001, max_iterations=1000000):  
    x = initial_guess

    for iteration in range(max_iterations):
        f_x = f(x)
        f_prime_x = f_prime(x)
        
        if abs(f_prime_x) < 1e-10:
            return None, iteration

        x_new = x - f_x / f_prime_x

        if np.abs(x_new - x) < tolerance:  
            return x_new, iteration + 1

        x = x_new

    return None, max_iterations

def get_equation_input():
    equation = simpledialog.askstring("Equation Input", "Enter the equation (use 'x' as the variable):")
    x_symbol = symbols('x')
    
    # Спрощення рівняння та створення функції для обчислень
    f_symbolic = simplify(equation)
    f = lambdify(x_symbol, f_symbolic, 'numpy')  # Перетворення рівняння на lambda-функцію

    # Визначення похідної від рівняння
    f_prime_symbolic = simplify(f_symbolic.diff(x_symbol))
    f_prime = lambdify(x_symbol, f_prime_symbolic, 'numpy')  # Перетворення похідної на lambda-функцію
    
    return f, f_prime

def transform_fx_to_gx_simple(f_eq, psi):
    """
    Перетворює рівняння f(x)=0 на x=g(x) за формулою: x = x - f(x).
    """
    x = symbols('x') 
    f = simplify(f_eq)
    
    try:
        # Використовуємо формулу g(x) = x - psi * f(x)
        g = (x - psi * f)
        g_lambda = lambdify(x, g, 'numpy')  # Повертаємо функцію g(x) у вигляді lambda

        return g_lambda
    except Exception as e:
        messagebox.showinfo("Error", f"Error in function transformation: {e}")
        return None

def nonlinear_simple_iteration(f_eq, initial_guess, psi, tolerance=0.0001, max_iterations=1000000):  
    g = transform_fx_to_gx_simple(f_eq, psi)  # Створюємо g(x) = x - f(x)
    x = round(initial_guess, 5)

    for iteration in range(max_iterations):
        x_new = g(x)
        x_new = round(x_new, 5)

        # print(f"Iteration {iteration}: x = {round(x, 5)}, g(x) = {x_new}")
        
        if np.abs(x_new - x) < tolerance:  
            return x_new, iteration + 1  # Повертаємо результат, якщо досягнута точність
        
        x = x_new
    
    return None, max_iterations

def get_matrix_input():
    n = simpledialog.askinteger("Input", "Enter the number of equations:")
    A = np.zeros((n, n))
    b = np.zeros(n)

    input_window = tk.Tk()
    input_window.title("Enter System of Equations")

    entry_widgets = [[tk.Entry(input_window) for _ in range(n + 1)] for _ in range(n)]
    for i in range(n):
        for j in range(n + 1):
            entry_widgets[i][j].grid(row=i, column=j, padx=5, pady=5)

    def get_coefficients():
        for i in range(n):
            for j in range(n + 1):
                if j < n:
                    A[i, j] = float(entry_widgets[i][j].get())
                else:
                    b[i] = float(entry_widgets[i][j].get())
        input_window.destroy()

    submit_button = tk.Button(input_window, text="Submit", command=get_coefficients)
    submit_button.grid(row=n, column=n // 2, pady=10)
    input_window.mainloop()

    return A, b

def display_solution(solution, iterations, method_name, result_window=None, precision=5):
    if solution is not None and all(sol is not None for sol in solution):
        n = len(solution)
        root_labels = [f'x{i + 1}' for i in range(n)]
        solution_str = "\n".join([f"{label}: {round(sol, precision)}" for label, sol in zip(root_labels, solution)])

        if iterations > 0:
            solution_str += f"\n\nIterations: {iterations}"

        if result_window is None:
            result_window = tk.Tk()
            result_window.title("Solution")

        result_label = tk.Label(result_window, text=solution_str)
        result_label.pack(pady=10)

        def close_result_window():
            if result_window.winfo_exists():
                result_window.destroy()

        back_to_method_selection_button = tk.Button(result_window, text="Back to Method Selection", command=lambda: [close_result_window(), show_method_selection()])
        back_to_method_selection_button.pack(pady=10)

        if not result_window.winfo_exists():
            result_window.mainloop()
    else:
        messagebox.showinfo("Solution", "Did not converge within the specified number of iterations or the determinant is zero.")

def get_evaluation_points():
    points_input = simpledialog.askstring("Evaluation Points", "Enter x values for evaluation (comma-separated):")
    if points_input is not None:
        return [float(x) for x in points_input.split(',')]
    else:
        return None

def newton_interpolation(data_points):
    n = len(data_points)
    x, y = zip(*data_points)

    # Initialize the divided differences table
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])

    # Build the symbolic polynomial
    x_sym = symbols('x')
    polynomial = divided_diff[0][0]
    term = 1
    
    for i in range(1, n):
        term *= (x_sym - x[i-1])
        polynomial += divided_diff[0][i] * term
    
    # Expand the polynomial to get a more readable form
    expanded_poly = expand(polynomial)
    
    def interpolated_polynomial(t):
        return float(expanded_poly.subs(x_sym, t))

    return interpolated_polynomial, latex(expanded_poly)

def lagrange_interpolation(data_points):
    x_vals, y_vals = zip(*data_points)
    n = len(x_vals)
    x_sym = symbols('x')

    def basis_poly(i):
        poly = 1
        for j in range(n):
            if i != j:
                poly *= (x_sym - x_vals[j]) / (x_vals[i] - x_vals[j])
        return poly

    # Construct the interpolated polynomial
    polynomial = sum(y * basis_poly(i) for i, y in enumerate(y_vals))
    expanded_poly = expand(polynomial)
    
    # Convert to a callable function
    interpolated_func = lambdify(x_sym, expanded_poly, 'numpy')

    return interpolated_func, latex(expanded_poly)


def get_lagrange_interpolation_data():
    data_input = simpledialog.askstring("Lagrange Interpolation", "Enter data points (x,y/x,y):")

    if data_input is not None:
        data_pairs = [tuple(map(float, pair.split(','))) for pair in data_input.split('/')]
        return data_pairs
    else:
        return None

def rectangular_integration(f, a, b, n):
    x_vals = np.linspace(a, b, n + 1)
    h = (b - a) / n
    integral_sum = h * np.sum(f(x_vals[:-1]))
    return integral_sum

# Метод трапецій
def trapezoidal_integration(f, a, b, n):
    x_vals = np.linspace(a, b, n + 1)
    h = (b - a) / n
    integral_sum = (h / 2) * (f(x_vals[0]) + 2 * np.sum(f(x_vals[1:-1])) + f(x_vals[-1]))
    return integral_sum

# Метод Сімпсона
def simpson_integration(f, a, b, n):
    if n % 2 == 1:  # Метод Сімпсона працює лише для парної кількості інтервалів
        n += 1
    x_vals = np.linspace(a, b, n + 1)
    h = (b - a) / n
    integral_sum = (h / 3) * (f(x_vals[0]) + 4 * np.sum(f(x_vals[1:-1:2])) + 2 * np.sum(f(x_vals[2:-2:2])) + f(x_vals[-1]))
    return integral_sum

def get_integration_input():
    equation = simpledialog.askstring("Integration Input", "Enter the integrand (use 'x' as the variable):")
    a = simpledialog.askfloat("Integration Input", "Enter the lower limit (a):")
    b = simpledialog.askfloat("Integration Input", "Enter the upper limit (b):")
    n = simpledialog.askinteger("Integration Input", "Enter the number of subintervals (n):")

    x_symbol = symbols('x')
    f = lambdify(x_symbol, simplify(equation), 'numpy')

    return f, a, b, n

"""def display_integration_result(result):
    if result is not None:
        messagebox.showinfo("Integration Result", f"The result of numerical integration is: {result}")
    else:
        messagebox.showinfo("Integration Result", "Integration failed. Check your input.")"""

def run_numerical_integration(method):
    f, a, b, n = get_integration_input()
    if method == "Rectangular Integration":
        result = rectangular_integration(f, a, b, n)
    elif method == "Trapezoidal Integration":
        result = trapezoidal_integration(f, a, b, n)
    elif method == "Simpson Integration":
        result = simpson_integration(f, a, b, n)

    # Plot the graph with red-outlined and red-transparent rectangles
    x_vals = np.linspace(a, b, n + 1)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, label='Function')

    for i in range(n):
        plt.gca().add_patch(plt.Rectangle((x_vals[i], 0), x_vals[i + 1] - x_vals[i], f(x_vals[i]),
                                         edgecolor='red', facecolor='red', alpha=0.3))

    if method == "Rectangular Integration":
        plt.title('Rectangular Integration')
    elif method == "Trapezoidal Integration":
        plt.title('Trapezoidal Integration')
    elif method == "Simpson Integration":
        plt.title('Simpson Integration')
    
    # plt.title('Numerical Integration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Display the graph
    plt.show()

    # Create a new result window
    result_window = tk.Tk()
    result_window.title("Numerical Integration Result")

    # Display the result
    result_label = tk.Label(result_window, text=f"The result of numerical integration is: {result:.5f}")
    result_label.pack(pady=10)

    # Button to return to the list of methods
    back_to_method_selection_button = tk.Button(result_window, text="Back to Method Selection", command=lambda: [result_window.destroy(), show_method_selection()])
    back_to_method_selection_button.pack(pady=10)

    result_window.mainloop()

def run_selected_method(method_name):
    if method_name == "Newton Method":
        f, f_prime = get_equation_input()
        initial_guess = simpledialog.askfloat("Newton Method", "Enter initial guess:")
        result, iterations = newton_method(f, f_prime, initial_guess)
        display_solution([result], iterations, method_name)
    elif method_name == "Simple Iteration":
        equation = simpledialog.askstring("Equation Input", "Enter the equation f(x) (use 'x' as the variable):")

        psi = simpledialog.askfloat("Simple Iteration", "Enter psi:")
        
        initial_guess = simpledialog.askfloat("Simple Iteration", "Enter initial guess:")
        
        result, iterations = nonlinear_simple_iteration(equation, initial_guess, psi)
        
        display_solution([result], iterations, method_name)

    elif method_name == "Newton Interpolation":
        data_points = get_interpolation_data()
        if data_points is not None:
            result = newton_interpolation(data_points)
            display_interpolation(result, data_points, "Newton Interpolation")

    elif method_name == "Lagrange Interpolation":
        data_points = get_lagrange_interpolation_data()
        if data_points is not None:
            result = lagrange_interpolation(data_points)
            display_interpolation(result, data_points, "Lagrange Interpolation")
    
    else:
        A, b = get_matrix_input()
        if method_name == "Gaussian Elimination":
            solution = gaussian_elimination(A, b)
            display_solution(solution, 0, method_name)
        else:
            messagebox.showinfo("Error", "Invalid method choice.")

def get_interpolation_data():
    data_input = simpledialog.askstring("Newton Interpolation", "Enter data points (x,y/x,y):")

    if data_input is not None:
        data_pairs = [tuple(map(float, pair.split(','))) for pair in data_input.split('/')]
        return data_pairs
    else:
        return None

def display_interpolation(interpolated_result, data_points, method_name):
    interpolated_polynomial, equation_latex = interpolated_result
    if interpolated_polynomial is not None:
        x_vals, y_vals = zip(*data_points)

        # Get evaluation points
        eval_points = get_evaluation_points()
        if eval_points is None:
            return

        # Create result window
        result_window = tk.Tk()
        result_window.title(f"{method_name} Result")
        
        # Create main frame with scrollbar
        main_frame = tk.Frame(result_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add canvas with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        # Configure scroll
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar elements
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create figure for equation
        fig_eq = Figure(figsize=(8, 1))
        ax_eq = fig_eq.add_subplot(111)
        ax_eq.text(0.5, 0.5, f"${equation_latex}$", 
                  fontsize=12, horizontalalignment='center',
                  verticalalignment='center')
        ax_eq.axis('off')
        
        # Add equation figure to scrollable frame
        canvas_eq = FigureCanvasTkAgg(fig_eq, master=scrollable_frame)
        canvas_eq.draw()
        canvas_eq.get_tk_widget().pack(pady=10)

        # Display evaluation points
        result_str = "Evaluation Points:\n"
        for x in eval_points:
            y = interpolated_polynomial(x)
            result_str += f"x: {round(x, 5)}, y: {round(y, 5)}\n"

        result_label = tk.Label(scrollable_frame, text=result_str)
        result_label.pack(pady=10)

        # Create figure for plot
        fig_plot = Figure(figsize=(8, 6))
        ax_plot = fig_plot.add_subplot(111)
        
        # Plot data
        x_range = np.linspace(min(x_vals), max(x_vals), 100)
        y_interp = [interpolated_polynomial(t) for t in x_range]
        
        ax_plot.plot(x_vals, y_vals, 'ro', label='Data Points')
        ax_plot.plot(x_range, y_interp, label='Interpolated Polynomial')
        
        eval_y = [interpolated_polynomial(x) for x in eval_points]
        ax_plot.scatter(eval_points, eval_y, color='blue', label='Evaluation Points')
        
        ax_plot.set_title(method_name)
        ax_plot.set_xlabel('x')
        ax_plot.set_ylabel('y')
        ax_plot.legend()
        ax_plot.grid(True)
        
        # Add plot to scrollable frame
        canvas_plot = FigureCanvasTkAgg(fig_plot, master=scrollable_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(pady=10)

        # Create frame for buttons in the main window (not in scrollable area)
        button_frame = tk.Frame(result_window)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # Add back button
        def back_to_menu():
            plt.close('all')  # Close all matplotlib figures
            result_window.destroy()
            show_method_selection()

        back_button = tk.Button(
            button_frame,
            text="Back to Method Selection",
            command=back_to_menu,
            width=20,
            height=2,
            relief=tk.RAISED,
            bd=3
        )
        back_button.pack(side=tk.LEFT, padx=5)

        # Add exit button
        def on_exit():
            plt.close('all')  # Close all matplotlib figures
            result_window.destroy()
            
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=on_exit,
            width=20,
            height=2,
            relief=tk.RAISED,
            bd=3
        )
        exit_button.pack(side=tk.LEFT, padx=5)

        # Set window size and position
        window_width = 800
        window_height = 700
        screen_width = result_window.winfo_screenwidth()
        screen_height = result_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        result_window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Make sure the window is resizable
        result_window.resizable(True, True)
        
        # Add mousewheel binding for scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
            
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        result_window.mainloop()
    else:
        messagebox.showinfo("Interpolation Result", "Interpolation failed. Check your input.")

def show_method_selection():
    methods = ["Gaussian Elimination", "Simple Iteration", "Newton Method", "Newton Interpolation", "Lagrange Interpolation",  "Rectangular Integration", "Trapezoidal Integration", "Simpson Integration"]

    method_selection_window = tk.Tk()
    method_selection_window.title("Choose Method")

    def on_method_click(method_name):
        method_selection_window.destroy()
        if method_name == "Rectangular Integration":
            run_numerical_integration(method_name)
        elif method_name == "Trapezoidal Integration":
            run_numerical_integration(method_name)
        elif method_name == "Simpson Integration":
            run_numerical_integration(method_name)
        else:
            run_selected_method(method_name)

    for method_name in methods:
        method_button = tk.Button(
            method_selection_window,
            text=method_name,
            command=lambda name=method_name: on_method_click(name)
        )
        method_button.pack(pady=10)

    method_selection_window.mainloop()

if __name__ == "__main__":
    show_method_selection()