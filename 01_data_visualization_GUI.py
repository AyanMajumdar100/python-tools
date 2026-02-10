import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import sys # Added to ensure clean exit

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# DATA PREPARATION
# ==============================
ages = [22, 25, 47, 52, 46, 56, 48, 30, 34, 29, 41, 38, 27, 33, 36]
income = [30000, 42000, 65000, 72000, 69000, 80000, 73000, 48000, 52000, 46000, 61000, 58000, 45000, 51000, 56000]
genders = ["Male", "Female", "Female", "Male", "Male", "Female", "Male",
           "Female", "Male", "Female", "Male", "Female", "Female", "Male", "Female"]

data = pd.DataFrame({
    "Age": ages,
    "Income": income,
    "Gender": genders
})

# ==============================
# PYTORCH MODEL TRAINING
# ==============================
def train_model():
    """Trains the PyTorch model and returns data for plotting."""
    X = torch.tensor(data["Age"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(data["Income"].values, dtype=torch.float32).view(-1, 1)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    epochs = 100
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    predicted = model(X).detach().numpy()
    return X.numpy(), y.numpy(), predicted

# ==============================
# GUI APPLICATION CLASS
# ==============================
class DataDashboardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI & Data Visualization Dashboard")
        self.geometry("1000x700")

        # ---------------------------------------------------------
        # FIX: Handle the Close Button to stop the app completely
        # ---------------------------------------------------------
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create Tab Control (Notebook)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Initialize Tabs
        self.create_histogram_tab()
        self.create_pie_tab()
        self.create_bar_tab()
        self.create_scatter_tab()
        self.create_pytorch_tab()

    def on_closing(self):
        """Clean up and close the application."""
        plt.close('all') # Close all matplotlib figures to free memory
        self.quit()      # Stop the mainloop
        self.destroy()   # Destroy the window
        sys.exit()       # Ensure the process terminates

    def embed_plot(self, fig, tab_name):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=tab_name)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def create_histogram_tab(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data["Age"], bins=5, color='skyblue', edgecolor='black')
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        self.embed_plot(fig, "Histogram")

    def create_pie_tab(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_counts = data["Gender"].value_counts()
        ax.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%", 
               colors=['lightcoral', 'lightblue'])
        ax.set_title("Gender Distribution")
        self.embed_plot(fig, "Pie Chart")

    def create_bar_tab(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_income = data.groupby("Gender")["Income"].mean()
        ax.bar(avg_income.index, avg_income.values, color=['lightcoral', 'lightblue'])
        ax.set_title("Average Income by Gender")
        ax.set_ylabel("Income ($)")
        self.embed_plot(fig, "Bar Graph")

    def create_scatter_tab(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data["Age"], data["Income"], color='purple')
        ax.set_title("Age vs Income")
        ax.set_xlabel("Age")
        ax.set_ylabel("Income")
        ax.grid(True, linestyle='--', alpha=0.6)
        self.embed_plot(fig, "Scatter Plot")

    def create_pytorch_tab(self):
        X, y, predicted = train_model()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X, y, color='purple', label='Actual Data')
        ax.plot(X, predicted, color='red', linewidth=2, label='PyTorch Prediction')
        ax.set_title("PyTorch Linear Regression Model")
        ax.set_xlabel("Age")
        ax.set_ylabel("Income")
        ax.legend()
        ax.grid(True)
        self.embed_plot(fig, "ML Model")

# ==============================
# MAIN ENTRY POINT
# ==============================
if __name__ == "__main__":
    app = DataDashboardApp()
    app.mainloop()