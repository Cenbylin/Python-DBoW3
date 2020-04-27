
#ifndef CMAKE_EXAMPLE_WRAPPER_DATABASE_H
#define CMAKE_EXAMPLE_WRAPPER_DATABASE_H

class Database {
public:
    Database(const std::string &path = std::string()) {
        if (path.empty())
            database = new DBoW3::Database();
        else
            database = new DBoW3::Database(path);
    }

    ~Database() {
        delete database;
    }

    void setVocabulary(const Vocabulary &vocabulary, bool use_di, int di_levels = 0) {
        database->setVocabulary(*vocabulary.vocabulary, use_di, di_levels);
    }

    unsigned int add(const cv::Mat &features) {
        return database->add(features, NULL, NULL);
    }

    std::vector<DBoW3::Result> query(const cv::Mat &features, int max_results = 1, int max_id = -1) {
        DBoW3::QueryResults results;
        database->query(features, results, max_results, max_id);
        return results;
    }

    void save(const std::string &filename) const {
        database->save(filename);
    }

    void load(const std::string &filename) {
        database->load(filename);
    }

    void loadVocabulary(const std::string &filename, bool use_di, int di_levels = 0) {
        DBoW3::Vocabulary voc;
        voc.load(filename);
        database->setVocabulary(voc, use_di, di_levels);
    }


private:
    DBoW3::Database *database;
};

#endif CMAKE_EXAMPLE_WRAPPER_DATABASE_H
