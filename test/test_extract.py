import unittest
from src.extract import (
    extract_affinities_from_file,
    extract_all_affinities,
    compile_multiple_software_results,
)


class TestDockingExtraction(unittest.TestCase):

    def test_extract_affinities_vina(self):
        result = extract_affinities_from_file("Data/vina/Log/90.txt")
        self.assertEqual(len(result["affinity"]), 322)
        self.assertEqual(result["affinity"][0], -9.982)
        self.assertEqual(result["affinity"][-1], 33.8)

    def test_extract_affinities_vina(self):
        result = extract_affinities_from_file("Data/smina/Log/90.txt")
        self.assertEqual(len(result["affinity"]), 2)
        self.assertEqual(result["affinity"][0], -10)
        self.assertEqual(result["affinity"][-1], -8.2)

    def test_extract_affinities_qvina(self):
        result = extract_affinities_from_file("Data/qvina/Log/90.txt")
        self.assertEqual(len(result["affinity"]), 2)
        self.assertEqual(result["affinity"][0], -10)
        self.assertEqual(result["affinity"][-1], -8.2)

    def test_extract_affinities_vina_gpu(self):
        result = extract_affinities_from_file("Data/Vina-GPU/Log/90.txt")
        self.assertEqual(len(result["affinity"]), 2)
        self.assertEqual(result["affinity"][0], -10)
        self.assertEqual(result["affinity"][-1], -8.2)

    def test_extract_affinities_vina_gpu2(self):
        result = extract_affinities_from_file("Data/Vina-GPU2/Log/90.txt")
        self.assertEqual(len(result["affinity"]), 3)
        self.assertEqual(result["affinity"][0], -10)
        self.assertEqual(result["affinity"][-1], -8.0)

    def test_extract_affinities_gnina(self):
        result = extract_affinities_from_file("Data/gnina/Log/90.txt", "gnina")
        self.assertEqual(len(result["affinity"]), 178)
        self.assertEqual(result["affinity"][0], -9.99)
        self.assertEqual(result["affinity"][-1], 191.59)
        self.assertEqual(result["cnn_pose_score"][0], 0.8833)
        self.assertEqual(result["cnn_pose_score"][-1], 0.0012)
        self.assertEqual(result["cnn_affinity"][0], 7.573)
        self.assertEqual(result["cnn_affinity"][-1], 3.418)

    def test_empty_file(self):
        result = extract_affinities_from_file("Data/emty.txt")
        self.assertEqual(result, {"affinity": []})

    def test_extract_path_all(self):
        result = extract_all_affinities("Data/vina/Log", software="vina", best=False)
        test_file = [value for value in result if value["ID"] == "90"][0]
        self.assertEqual(len(test_file["vina"]), 322)
        self.assertEqual(test_file["vina"][0], -9.982)
        self.assertEqual(test_file["vina"][-1], 33.8)

    def test_extract_path_best(self):
        result = extract_all_affinities("Data/vina/Log", software="vina", best=True)
        test_file = [value for value in result if value["ID"] == "90"][0]
        self.assertTrue(type(test_file["vina"]), int)
        self.assertEqual(test_file["vina"], -9.982)

    def test_compile_multiple_softwares(self):
        software_list = ["vina", "smina", "qvina"]
        base_directory = "Data"
        best = True
        df = compile_multiple_software_results(software_list, base_directory, best)
        self.assertEqual(df["vina_affinity"][0], -8.225)
        self.assertEqual(df["smina_affinity"][0], -8.6)
        self.assertEqual(df["qvina_affinity"][0], -8.2)


if __name__ == "__main__":
    unittest.main()
