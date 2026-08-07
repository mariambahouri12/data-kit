import io

class FileAdapter:

    def __init__(self, upload_file):

        self.name = upload_file.filename

        content = upload_file.file.read()

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

        self.file = io.StringIO(text)


    def read(self, *args):
        return self.file.read(*args)


    def seek(self, *args):
        return self.file.seek(*args)