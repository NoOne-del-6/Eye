from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QFileDialog, QVBoxLayout, QWidget, QPushButton
import pandas as pd
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import numpy as np

class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(pd.DataFrame, int)  # Signal to pass loaded data and batch number back to UI

    def __init__(self, file_path, chunk_size=1000):
        super().__init__()
        self.file_path = file_path
        self.chunk_size = chunk_size

    def run(self):
        # Read CSV file in chunks and emit the data to the main thread
        try:
            # Load data in chunks
            chunk_iter = pd.read_csv(self.file_path, chunksize=self.chunk_size)
            for chunk in chunk_iter:
                self.data_loaded.emit(chunk, len(chunk))  # Emit the chunk of data to update the table
        except Exception as e:
            print(f"Error loading data: {e}")


class TableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self.data = data if data is not None else pd.DataFrame()

    def rowCount(self, parent=None):
        return len(self.data)

    def columnCount(self, parent=None):
        return len(self.data.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self.data.iloc[index.row(), index.column()])
        return None

    def setData(self, data):
        self.data = data
        self.layoutChanged.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized Data Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.model = TableModel()  # Create model instance for QTableView
        self.table_view = QTableView(self)
        self.table_view.setModel(self.model)
        self.table_view.setSortingEnabled(True)  # Enable sorting if necessary

        # Layout and widgets
        layout = QVBoxLayout()

        self.load_button = QPushButton("Load CSV File", self)
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        layout.addWidget(self.table_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.data_thread = None

    def load_csv(self):
        # Open file dialog to select CSV file
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.load_button.setEnabled(False)  # Disable button during loading
            self.data_thread = DataLoaderThread(file_path, chunk_size=1000)
            self.data_thread.data_loaded.connect(self.update_table_data)
            self.data_thread.start()

    def update_table_data(self, data, chunk_size):
        # Append the new chunk of data to the model
        current_data = self.model.data  # Current data in the table
        updated_data = pd.concat([current_data, data], ignore_index=True)  # Append new data
        self.model.setData(updated_data)

        # Update button and state after the loading
        self.load_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
