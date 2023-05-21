from recognition import load_model, doc_bang_diem, recognize_file, recognize_folder

if __name__ == "__main__":
    model = load_model()

    file_path = ".\\Data\\Bangdiem5\\GT2\\GT2 1\\1.jpg"
    output_path = ".\\Output"
    recognize_file(model, file_path, output_path)