
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

// Python
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Needed for std::vector conversion


namespace py = pybind11;

using namespace DBoW2;
using namespace std;


class ImageProcessor {
public:
    ImageProcessor(const std::string& path_voc, const std::string& path_db) : mVoc(), mDb() {

        if (!path_voc.empty()) {
            mVoc.load(path_voc);
        }

        if (!path_db.empty()) {
            mDb.load(path_db);
        }

        mOrbExtractor = cv::ORB::create();
    }  // Constructor

    // Method to process an image
    std::vector<int> processImage(const py::array_t<unsigned char>& input_array, int n_matches) {

        py::buffer_info buf_info = input_array.request();
        int height = (int) buf_info.shape[0];
        int width = (int) buf_info.shape[1];
        auto* ptr = static_cast<unsigned char*>(buf_info.ptr);

        cv::Mat image(height, width, CV_8UC1, ptr);

        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        mOrbExtractor->detectAndCompute(image, mask, keypoints, descriptors);

        vector<cv::Mat> features;
        changeStructure(descriptors, features);

        QueryResults ret;
        mDb.query(features, ret, n_matches);

        int n_found = ret.size();
        std::vector<int> results;

        if (n_found > 0) {
            results.resize(2 * n_found);
        }
        else {
            return results;
        }
        
        for (size_t i = 0; i < n_matches && i < ret.size(); i++) {
            results[i] = ret[i].Id;
            results[i + n_found] = static_cast<int>(ret[i].Score * 100);
        }
        
        return results;
    }

    static void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) {

        out.resize(plain.rows);

        for(int i = 0; i < plain.rows; ++i)
        {
            out[i] = plain.row(i);
        }
    }

    static void testVocCreation(const vector<vector<cv::Mat>> &features, const std::string& save_path, int nImages) {

        // branching factor and depth levels
        const int k = 9;
        const int L = 3;
        const WeightingType weight = TF_IDF;
        const ScoringType scoring = L1_NORM;

        OrbVocabulary voc(k, L, weight, scoring);

        cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
        voc.create(features);
        cout << "... done!" << endl;

        cout << "Vocabulary information: " << endl
             << voc << endl << endl;

        // save the vocabulary to disk
        cout << endl << "Saving vocabulary..." << endl;
        voc.save(save_path + "/small_voc.yml.gz");
        cout << "Done" << endl;
    }

    static void testDatabase(const vector<vector<cv::Mat>> &features, const std::string& save_path, int nImages) {

        cout << "Creating a small database..." << endl;

        // load the vocabulary from disk
        OrbVocabulary voc(save_path + "/small_voc.yml.gz");

        OrbDatabase db(voc, false, 0); // false = do not use direct index
        // (so ignore the last param)
        // The direct index is useful if we want to retrieve the features that
        // belong to some vocabulary node.
        // db creates a copy of the vocabulary, we may get rid of "voc" now

        // add images to the database
        for(int i = 0; i < nImages; i++)
        {
            db.add(features[i]);
        }

        cout << "... done!" << endl;

        cout << "Database information: " << endl << db << endl;

        // we can save the database. The created file includes the vocabulary
        // and the entries added
        const std::string db_path = save_path + "/small_db.yml.gz";
        cout << "Saving database..." << endl;
        db.save(db_path);
        cout << "... done!" << endl;

        // once saved, we can load it again
        cout << "Retrieving database once again..." << endl;
        OrbDatabase db2(db_path);
        cout << "... done! This is: " << endl << db2 << endl;
    }

    static void loadFeatures(vector<vector<cv::Mat>> &features, const std::string& images_path, int nImages) {

        features.clear();
        features.reserve(nImages);

        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        cout << "Extracting ORB features..." << endl;
        for(int i = 0; i < nImages; ++i)
        {
            stringstream ss;
            ss << images_path << "/image" << i << ".png";

            cv::Mat image = cv::imread(ss.str(), 0);
            cv::Mat mask;
            vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;

            orb->detectAndCompute(image, mask, keypoints, descriptors);

            features.emplace_back();
            changeStructure(descriptors, features.back());
        }
    }

    void createVocAndDb(const std::string& images_path, const std::string& save_path, int n_images) {

        vector<vector<cv::Mat >> features;
        loadFeatures(features, images_path, n_images);

        testVocCreation(features, save_path, n_images);

        //wait();

        testDatabase(features, save_path, n_images);
    }

    static void wait() {
        cout << endl << "Press enter to continue" << endl;
        getchar();
    }

private:
    OrbVocabulary mVoc;
    OrbDatabase mDb;
    cv::Ptr<cv::ORB> mOrbExtractor;
//    int mnImages;
};

// Bind the class and functions with pybind11
PYBIND11_MODULE(imageproc, m) {
    py::class_<ImageProcessor>(m, "ImageProcessor")
        .def(py::init<const std::string&, const std::string&>())  // Constructor with two strings
        .def("process_image", &ImageProcessor::processImage, py::arg("input_array"), py::arg("n_matches"))  // New function with int argument
        .def("create_voc_db", &ImageProcessor::createVocAndDb, py::arg("images_path"), py::arg("save_path"), py::arg("n_images"));
}

