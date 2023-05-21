from cv2 import split
from GUI import Ui_MainWindow
import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QTableView
from PyQt5.QtGui import QIcon, QPixmap
from recognition import load_model, doc_bang_diem
import os
from recognition import doc_bang_diem
import pandas as pd
from PIL import Image, ImageOps
from PyQt5.QtCore import QAbstractTableModel, Qt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnBrowseInput.clicked.connect(self.click_btnBrowseInput)
        self.btnBrowseOutput.clicked.connect(self.click_btnBrowseOutput)
        self.btnRun.clicked.connect(self.click_btnRun)
        self.checkbox_file.stateChanged.connect(self.check_checkbox_file)
        self.checkbox_folder.stateChanged.connect(self.check_checkbox_folder)


        self.input_path = ""
        self.output_path = ""
        self.list_file_path = []
        self.model = load_model()

    def check_checkbox_file(self):
        if self.checkbox_file.isChecked() is True:
            if self.checkbox_folder.isChecked():
                self.checkbox_folder.setChecked(False)
        else:
            if self.checkbox_folder.isChecked() is False:
                self.checkbox_folder.setChecked(True)
    

    def check_checkbox_folder(self):
        if self.checkbox_folder.isChecked() is True:
            if self.checkbox_file.isChecked():
                self.checkbox_file.setChecked(False)
        else:
            if self.checkbox_file.isChecked() is False:
                self.checkbox_file.setChecked(True)


    def click_btnBrowseInput(self):
        if self.checkbox_file.isChecked() is True and self.checkbox_folder.isChecked() is False:
            self.input_path = QtWidgets.QFileDialog.getOpenFileName(self,"Choose input file")[0]
            self.txtInputPath.setText(self.input_path)
            self.list_file_path.append(self.input_path)
        if self.checkbox_file.isChecked() is False and self.checkbox_folder.isChecked() is True:
            self.input_path = QtWidgets.QFileDialog.getExistingDirectory(self,"Choose input directory")
            self.txtInputPath.setText(self.input_path)
            for file_name in os.listdir(self.input_path):
                self.list_file_path.append(os.path.join(self.input_path, file_name))
        for file_url in self.list_file_path:
            img = QPixmap(file_url)
            img = img.scaled(451, 631)
            self.lbl_visualize.setPixmap(img)



    def click_btnBrowseOutput(self):
        self.output_path = QtWidgets.QFileDialog.getExistingDirectory(self,"Choose output directory")
        self.txtOutputPath.setText(self.output_path)
    

    def click_btnRun(self):
        for file_path in self.list_file_path:
            excel_filename, df = doc_bang_diem(file_path, self.model)
            print("file: ", excel_filename)
            print(df)

            self.lbl_malop.setText("Mã lớp: " + excel_filename.split(".")[0])
            model = pandasModel(df)
            self.tableview_dataframe.setModel(model)
            self.tableview_dataframe.show()

            # if os.path.exists(os.path.join(self.output_path, excel_filename)) is True:
            #     old_df = pd.read_excel(os.path.join(self.output_path, excel_filename))
            #     old_df = pd.DataFrame(old_df)
            #     new_df = pd.concat([old_df, df])
            #     new_df.to_excel(os.path.join(self.output_path, excel_filename), index=False)
            # else:
            #     df.to_excel(os.path.join(self.output_path, excel_filename), index=False)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()