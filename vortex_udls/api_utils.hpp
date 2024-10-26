#pragma once
#include <string>
#include <vector>

namespace api_utils {

std::string get_embedding(const std::string &query, const std::string &model, const std::string &api_key, int dim);
std::string run_gpt4o_mini(const std::string &query, const std::vector<std::string> &top_docs, const std::string &model, const std::string &api_key);

}
