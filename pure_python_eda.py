import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PurePythonEDA:
    def __init__(self, root):
        self.root = root
        self.root.title("Pure Python EDA Visualizer")
        self.root.geometry("1100x800")
        self.root.configure(bg="#f0f2f5")

        self.df = None
        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Python EDA (Matplotlib + Seaborn)", font=("Arial", 20, "bold"), bg="#f0f2f5", fg="#1e293b", pady=20)
        header.pack()

        # Sidebar Frame
        self.sidebar = tk.Frame(self.root, width=300, bg="#ffffff", padx=20, pady=20, relief=tk.RIDGE, borderwidth=1)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Upload Button
        self.upload_btn = tk.Button(self.sidebar, text="📂 Load CSV File", command=self.load_file, bg="#3b82f6", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10)
        self.upload_btn.pack(fill=tk.X, pady=(0, 20))

        # Config Controls (Hidden until file loaded)
        self.config_frame = tk.Frame(self.sidebar, bg="#ffffff")
        
        tk.Label(self.config_frame, text="Select Plot Type:", bg="#ffffff", font=("Arial", 10)).pack(anchor=tk.W)
        self.plot_type = ttk.Combobox(self.config_frame, values=[
            "Bar Plot", "Scatter Plot", "Line Plot", "Histogram", "Box Plot", "Violin Plot", "Correlation Heatmap"
        ])
        self.plot_type.pack(fill=tk.X, pady=(0, 15))
        self.plot_type.bind("<<ComboboxSelected>>", self.update_column_options)

        self.col1_label = tk.Label(self.config_frame, text="X Axis / Column:", bg="#ffffff", font=("Arial", 10))
        self.col1_label.pack(anchor=tk.W)
        self.col1_box = ttk.Combobox(self.config_frame)
        self.col1_box.pack(fill=tk.X, pady=(0, 15))

        self.col2_label = tk.Label(self.config_frame, text="Y Axis / Value (Optional):", bg="#ffffff", font=("Arial", 10))
        self.col2_label.pack(anchor=tk.W)
        self.col2_box = ttk.Combobox(self.config_frame)
        self.col2_box.pack(fill=tk.X, pady=(0, 15))

        self.visualize_btn = tk.Button(self.config_frame, text="📉 Generate Plot", command=self.generate_plot, bg="#10b981", fg="white", font=("Arial", 10, "bold"), pady=10)
        self.visualize_btn.pack(fill=tk.X, pady=(20, 0))

        # Main Plot Area
        self.plot_area = tk.Frame(self.root, bg="#f0f2f5")
        self.plot_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        self.empty_label = tk.Label(self.plot_area, text="Please load a CSV to start", font=("Arial", 12), bg="#f0f2f5", fg="#64748b")
        self.empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.config_frame.pack(fill=tk.X)
                self.empty_label.config(text=f"Loaded: {file_path.split('/')[-1]}")
                
                cols = self.df.columns.tolist()
                self.col1_box['values'] = cols
                self.col2_box['values'] = cols
                
                messagebox.showinfo("Success", f"CSV Loaded Successfully!\nTotal Rows: {len(self.df)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read CSV: {e}")

    def update_column_options(self, event):
        p_type = self.plot_type.get()
        if p_type == "Correlation Heatmap":
            self.col1_box.config(state="disabled")
            self.col2_box.config(state="disabled")
        elif p_type in ["Histogram"]:
            self.col1_box.config(state="normal")
            self.col2_box.config(state="disabled")
        else:
            self.col1_box.config(state="normal")
            self.col2_box.config(state="normal")

    def generate_plot(self):
        if self.df is None: return

        p_type = self.plot_type.get()
        c1 = self.col1_box.get()
        c2 = self.col2_box.get()

        # Clear previous plot
        for widget in self.plot_area.winfo_children():
            widget.destroy()

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))

        try:
            if p_type == "Bar Plot":
                sns.barplot(data=self.df.head(20), x=c1, y=c2, ax=ax)
            elif p_type == "Scatter Plot":
                sns.scatterplot(data=self.df, x=c1, y=c2, ax=ax)
            elif p_type == "Line Plot":
                sns.lineplot(data=self.df.head(100), x=c1, y=c2, ax=ax)
            elif p_type == "Histogram":
                sns.histplot(self.df[c1], kde=True, ax=ax)
            elif p_type == "Box Plot":
                sns.boxplot(data=self.df, x=c1, y=c2, ax=ax)
            elif p_type == "Violin Plot":
                sns.violinplot(data=self.df, x=c1, y=c2, ax=ax)
            elif p_type == "Correlation Heatmap":
                num_df = self.df.select_dtypes(include=['number'])
                sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PurePythonEDA(root)
    root.mainloop()
