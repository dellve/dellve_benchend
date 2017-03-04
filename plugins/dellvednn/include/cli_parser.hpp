#ifndef DELLVE_CLIPARSER_H_
#define DELLVE_CLIPARSER_H_

#include <vector>

class CLIParser{
private:
    std::vector<std::string> tokens_;
    std::string empty_string_;
    std::string dummy_string_;

    std::vector<std::tuple<std::string, std::string, bool, std::string*>> options_;
    std::string file_;

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
            std::make_tuple("-p", "Pass in Problem Set", true, &file_)
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
                exit(0);
            }
        }
    }
};

#endif // DELLVE_CLI_PARSER_H_

