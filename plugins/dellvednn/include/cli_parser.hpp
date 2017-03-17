#ifndef DELLVE_CLIPARSER_H_
#define DELLVE_CLIPARSER_H_

#include <vector>

/*
 * Temporary Parser in cpp. 
 */
class CLIParser{
private:
    std::vector<std::string> tokens_;
    std::string empty_string_;
    std::string dummy_string_;

    std::vector<std::tuple<std::string, std::string, bool, std::string*>> options_;
    std::string file_;
    std::string num_runs_;
    std::string gpus_;

public:
    CLIParser (int &argc, char **argv) {
        parseTokens(argc, argv);
        initializeOptions();
        displayHelp();
        parseOptions();
    }

    std::string getProblemSetFile(void) {
        return file_;
    }

    // Raw
    int getNumRuns(void) {
        if(num_runs_.empty()) {
            return 50;
        } else {
            return atoi(num_runs_.c_str());
        }
    }
    
    // Maybe better way to do it? x_x parsing in c zzz
    std::vector<int> getGpus(void) {
        if(gpus_.empty()) { // Default = 1
            return {1};
        } else {
            std::string delim = ","; // Look for comma separated ints
            std::vector<int> result = {};
            auto start = 0U;
            auto end = gpus_.find(delim);
            while(end != std::string::npos) {
                result.push_back(atoi(gpus_.substr(start, end-start).c_str()));
                start = end + delim.length();
                end = gpus_.find(delim, start);
            }
            if(gpus_.c_str()[gpus_.size()-1] != ',') { // if last one isn't a comma, add it in
                result.push_back(atoi(gpus_.substr(start, end-start).c_str()));
            } 
            return result;
        }
    }

private:
    void parseTokens(int &argc, char **argv) {
        for(int i = 1; i < argc; ++i) {
            tokens_.push_back(std::string(argv[i]));
        }
    }

    const std::string& getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(this->tokens_.begin(), this->tokens_.end(), option);
        if (itr != this->tokens_.end() && ++itr != this->tokens_.end()){
            return *itr;
        }
        return empty_string_;
    }

    bool cmdOptionExists(const std::string &option) const{
        return std::find(this->tokens_.begin(), this->tokens_.end(), option)
               != this->tokens_.end();
    }
    
    void initializeOptions(void) {
        options_ = {
            std::make_tuple("-h", "Display this Help Message", false, &dummy_string_),
            std::make_tuple("-p", "Problem Set to run through", true, &file_),
            std::make_tuple("-r", "Number of runs per problem set. Default 50", false, &num_runs_),
            std::make_tuple("-g", "Specify GPUs to run as comma separated. Default 1", false, &gpus_)
        };
    }

    void displayHelp(void) {
        if(cmdOptionExists("-h")) {
            printf("Available Options for DELLve Benchmarks:\n\n");
            std::string option;
            std::string desc;
            std::string* ptr;
            bool req;    
            for( auto &i : options_ ) {
                std::tie(option, desc, req, ptr) = i;
                printf("%s\t%s\n", option.c_str(), desc.c_str());
            }
            exit(0);
        }
    }

    void parseOptions(void) {
        std::string option;
        std::string desc;
        std::string* ptr;
        bool req;    
        for( auto &i : options_ ) {
            std::tie(option, desc, req, ptr) = i;
            *ptr = getCmdOption(option);
            if(req && ptr->empty()) {
                printf("Parse error: %s, specify through %s\n", desc.c_str(), option.c_str());
                printf("Run with -h option to see all options\n");
                exit(0);
            }
        }
    }
};

#endif // DELLVE_CLI_PARSER_H_

