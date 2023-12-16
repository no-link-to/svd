from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class SVDWorker:
    def __init__(self, img):
        self.image_path = img
        self.converted_grey = None
        self.image_matrix = None
        self.available_values = [100, 90, 80, 70, 60, 50, 40, 10, 5, 1]
        self.result = []

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value

    @property
    def converted_grey(self):
        return self._converted_grey

    @converted_grey.setter
    def converted_grey(self, value):
        self._converted_grey = value

    @property
    def image_matrix(self):
        return self._image_matrix

    @image_matrix.setter
    def image_matrix(self, value):
        self._image_matrix = value

    @property
    def available_values(self):
        return self._available_values

    @available_values.setter
    def available_values(self, value):
        self._available_values = value

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value):
        self._result = value

    def get_matrix_size(self):
        return self.image_matrix.shape

    def convert_image_to_grey(self):
        self.converted_grey = Image.open(self.image_path).convert('L')

    def create_image_matrix(self):
        if self.converted_grey:
            self.image_matrix = np.array(self.converted_grey)

    def get_compression(self, k):
        m, n = self.image_matrix.shape
        value_compr = k * (m + n + 1)
        value_source = m * n
        return value_compr / value_source * 100

    def calculate_results(self):
        local_result = []
        for value in self.available_values:
            # SVD разложение
            u, s, vh = np.linalg.svd(self.image_matrix, full_matrices=False)
            # Создание матрицы с меньшим количеством сингулярных чисел
            local_result.append(u[:, :value] @ np.diag(s[:value]) @ vh[:value, :])
        self.result = local_result

    def draw_figures(self):
        for i in range(len(self.result)):
            plt.figure(figsize=(10, 6))
            plt.imshow(self.result[i], cmap='gray')
            plt.title(f'k = {self.available_values[i]}')
            plt.axis('off')
            plt.show()

    def draw_grey_source_figure(self):
        plt.figure(figsize=(10, 6))
        plt.imshow(self.image_matrix, cmap='gray')
        plt.title(f'Серая собачка')
        plt.axis('off')
        plt.show()

    def run(self):
        self.convert_image_to_grey()
        self.create_image_matrix()
        self.draw_grey_source_figure()
        self.calculate_results()
        self.draw_figures()


if __name__ == '__main__':
    svd_worker = SVDWorker("dog_img.jpg")
    svd_worker.run()
