/*=======================================================================
 * Copyright 2020-2023 Enflame. All Rights Reserved.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *=======================================================================
 */

#ifndef ARGPARSE_HPP
#define ARGPARSE_HPP

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace arg {

static inline bool not_space(int ch) { return !std::isspace(ch); }
static inline void ltrim(std::string &s, bool (*f)(int) = not_space) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), f));
}
static inline void rtrim(std::string &s, bool (*f)(int) = not_space) {
  s.erase(std::find_if(s.rbegin(), s.rend(), f).base(), s.end());
}
static inline void trim(std::string &s, bool (*f)(int) = not_space) {
  ltrim(s, f);
  rtrim(s, f);
}
static inline std::string ltrim_copy(std::string s,
                                     bool (*f)(int) = not_space) {
  ltrim(s, f);
  return s;
}
static inline std::string rtrim_copy(std::string s,
                                     bool (*f)(int) = not_space) {
  rtrim(s, f);
  return s;
}
static inline std::string trim_copy(std::string s, bool (*f)(int) = not_space) {
  trim(s, f);
  return s;
}
template <typename InputIt>
static inline std::string join(InputIt begin, InputIt end,
                               const std::string &separator = " ") {
  std::ostringstream ss;
  if (begin != end) {
    ss << *begin++;
  }
  while (begin != end) {
    ss << separator;
    ss << *begin++;
  }
  return ss.str();
}
static inline bool is_number(const std::string &arg) {
  std::istringstream iss(arg);
  float f;
  iss >> std::noskipws >> f;
  return iss.eof() && !iss.fail();
}

static inline int find_equal(const std::string &s) {
  for (size_t i = 0; i < s.length(); ++i) {

    if (std::ispunct(static_cast<int>(s[i]))) {
      if (s[i] == '=') {
        return static_cast<int>(i);
      }
      if (s[i] == '_' || s[i] == '-') {
        continue;
      }
      return -1;
    }
  }
  return -1;
}

static inline size_t find_name_end(const std::string &s) {
  size_t i;
  for (i = 0; i < s.length(); ++i) {
    if (std::ispunct(static_cast<int>(s[i]))) {
      if (s[i] == '-' || s[i] == '=') {
        break;
      }
    }
  }
  return i;
}

namespace is_vector_impl {
template <typename T> struct IsVector : std::false_type {};
template <typename... Args>
struct IsVector<std::vector<Args...>> : std::true_type {};
} // namespace is_vector_impl

template <typename T> struct IsVector {
  static constexpr bool const value =
      is_vector_impl::IsVector<typename std::decay<T>::type>::value;
};

class ArgParser {
private:
public:
  class Argument;

  class Result {
  public:
    Result() = default;
    explicit Result(std::string err) noexcept
        : error_(true), what_(std::move(err)) {}

    explicit operator bool() const { return error_; }

    friend std::ostream &operator<<(std::ostream &os, const Result &r);

    const std::string &what() const { return what_; }

  private:
    bool error_{false};
    std::string what_{};
  };

  class Argument {
  public:
    enum Position : int { kLast = -1, kDontCare = -2 };
    enum Count : int { kAny = -1 };

    Argument &name(const std::string &name) {
      names_.push_back(name);
      return *this;
    }

    Argument &names(std::vector<std::string> names) {
      names_.insert(names_.end(), names.begin(), names.end());
      return *this;
    }

    Argument &description(const std::string &description) {
      desc_ = description;
      return *this;
    }

    Argument &required(bool req) {
      required_ = req;
      return *this;
    }

    Argument &position(int position) {
      if (position != Position::kLast) {
        position_ = position + 1;
      } else {
        position_ = position;
      }
      return *this;
    }

    Argument &count(int count) {
      count_ = count;
      return *this;
    }

    bool found() const { return found_; }

    template <typename T>
    typename std::enable_if<IsVector<T>::value, T>::type get() {
      T t = T();
      typename T::value_type vt;
      for (auto &s : values_) {
        std::istringstream in(s);
        in >> vt;
        t.push_back(vt);
      }
      return t;
    }

    template <typename T>
    typename std::enable_if<!IsVector<T>::value, T>::type get() {
      std::istringstream in(get<std::string>());
      T t = T();
      in >> t >> std::ws;
      return t;
    }

  private:
    Argument(const std::string &name, std::string desc, bool required = false)
        : desc_(std::move(desc)), required_(required) {
      names_.push_back(name);
    }

    Argument() = default;

    friend class ArgParser;
    int position_{Position::kDontCare};
    int count_{Count::kAny};
    std::vector<std::string> names_{};
    std::string desc_{};
    bool found_{false};
    bool required_{false};
    int index_{-1};

    std::vector<std::string> values_{};
  };

  ArgParser(std::string bin, std::string desc)
      : bin_(std::move(bin)), desc_(std::move(desc)) {}

  Argument &add() {
    arguments_.push_back({});
    arguments_.back().index_ = static_cast<int>(arguments_.size()) - 1;
    return arguments_.back();
  }

  Argument &add(const std::string &name, const std::string &long_name,
                const std::string &desc, const bool required = false) {
    arguments_.push_back(Argument(name, desc, required));
    arguments_.back().names_.push_back(long_name);
    arguments_.back().index_ = static_cast<int>(arguments_.size()) - 1;
    return arguments_.back();
  }

  Argument &add(const std::string &name, const std::string &desc,
                const bool required = false) {
    arguments_.push_back(Argument(name, desc, required));
    arguments_.back().index_ = static_cast<int>(arguments_.size()) - 1;
    return arguments_.back();
  }

  void help(size_t count = 0, size_t page = 0) {
    if (page * count > arguments_.size()) {
      return;
    }
    if (page == 0) {
      std::cout << "Usage: " << bin_;
      if (positional_arguments_.empty()) {
        std::cout << " [options...]" << std::endl;
      } else {
        int current = 1;
        for (auto &v : positional_arguments_) {
          if (v.first != Argument::Position::kLast) {
            for (; current < v.first; current++) {
              std::cout << " [" << current << "]";
            }
            std::cout
                << " ["
                << ltrim_copy(
                       arguments_[static_cast<size_t>(v.second)].names_[0],
                       [](int c) -> bool { return c != static_cast<int>('-'); })
                << "]";
          }
        }
        auto it = positional_arguments_.find(Argument::Position::kLast);
        if (it == positional_arguments_.end()) {
          std::cout << " [options...]";
        } else {
          std::cout
              << " [options...] ["
              << ltrim_copy(
                     arguments_[static_cast<size_t>(it->second)].names_[0],
                     [](int c) -> bool { return c != static_cast<int>('-'); })
              << "]";
        }
        std::cout << std::endl;
      }
      std::cout << "Options:" << std::endl;
    }
    if (count == 0) {
      page = 0;
      count = arguments_.size();
    }
    for (size_t i = page * count;
         i < std::min<size_t>(page * count + count, arguments_.size()); i++) {
      Argument &a = arguments_[i];
      std::string name = a.names_[0];
      for (size_t n = 1; n < a.names_.size(); ++n) {
        name.append(", " + a.names_[n]);
      }
      std::cout << "    " << std::setw(23) << std::left << name << std::setw(23)
                << a.desc_;
      if (!a.required_) {
        std::cout << " (Optional)";
      }
      std::cout << std::endl;
    }
  }

  Result parse(int argc, const char *argv[]) {
    Result err;
    if (argc > 1) {
      // build name map
      for (auto &a : arguments_) {
        for (auto &n : a.names_) {
          std::string name = ltrim_copy(
              n, [](int c) -> bool { return c != static_cast<int>('-'); });
          if (name_map_.find(name) != name_map_.end()) {
            return Result("Duplicate of argument name: " + n);
          }
          name_map_[name] = a.index_;
        }
        if (a.position_ >= 0 || a.position_ == Argument::Position::kLast) {
          positional_arguments_[a.position_] = a.index_;
        }
      }
      if (err) {
        return err;
      }

      // parse
      std::string current_arg;
      size_t arg_len;
      for (int argv_index = 1; argv_index < argc; ++argv_index) {
        current_arg = std::string(argv[argv_index]);
        arg_len = current_arg.length();
        if (arg_len == 0) {
          continue;
        }
        if (help_enabled_ && (current_arg == "-h" || current_arg == "--help")) {
          arguments_[static_cast<size_t>(name_map_["help"])].found_ = true;
        } else if (argv_index == argc - 1 &&
                   positional_arguments_.find(Argument::Position::kLast) !=
                       positional_arguments_.end()) {
          err = end_argument();
          Result b = err;
          err = add_value(current_arg, Argument::Position::kLast);
          if (b) {
            return b;
          }
          if (err) {
            return err;
          }
        } else if (arg_len >= 2 &&
                   !is_number(current_arg)) { // ignores the case if
                                              // the arg is just a -
          // look for -a (short) or --arg (long) args
          if (current_arg[0] == '-') {
            err = end_argument();
            if (err) {
              return err;
            }
            // look for --arg (long) args
            if (current_arg[1] == '-') {
              err = begin_argument(current_arg.substr(2), true, argv_index);
              if (err) {
                return err;
              }
            } else { // short args
              err = begin_argument(current_arg.substr(1), false, argv_index);
              if (err) {
                return err;
              }
            }
          } else { // argument value
            err = add_value(current_arg, argv_index);
            if (err) {
              return err;
            }
          }
        } else { // argument value
          err = add_value(current_arg, argv_index);
          if (err) {
            return err;
          }
        }
      }
    }
    if (help_enabled_ && exists("help")) {
      return Result();
    }
    err = end_argument();
    if (err) {
      return err;
    }
    for (auto &p : positional_arguments_) {
      Argument &a = arguments_[static_cast<size_t>(p.second)];
      if (!a.values_.empty() && a.values_[0][0] == '-') {
        std::string name = ltrim_copy(a.values_[0], [](int c) -> bool {
          return c != static_cast<int>('-');
        });
        if (name_map_.find(name) != name_map_.end()) {
          if (a.position_ == Argument::Position::kLast) {
            return Result("expected at the end, but argument " + a.values_[0] +
                          " found.");
          }
          return Result("argument expected in position " +
                        std::to_string(a.position_) + ", but argument " +
                        a.values_[0] + " found.");
        }
      }
    }
    for (auto &a : arguments_) {
      if (a.required_ && !a.found_) {
        return Result("Required argument not found: " + a.names_[0]);
      }
      if (a.position_ >= 0 && argc >= a.position_ && !a.found_) {
        return Result("Argument " + a.names_[0] + " expected in position " +
                      std::to_string(a.position_));
      }
    }
    return Result();
  }

  void enable_help() {
    add("-h", "--help", "Shows this page", false);
    help_enabled_ = true;
  }

  bool exists(const std::string &name) const {
    std::string n = ltrim_copy(
        name, [](int c) -> bool { return c != static_cast<int>('-'); });
    auto it = name_map_.find(n);
    if (it != name_map_.end()) {
      return arguments_[static_cast<size_t>(it->second)].found_;
    }
    return false;
  }

  template <typename T> T get(const std::string &name) {
    auto t = name_map_.find(name);
    if (t != name_map_.end()) {
      return arguments_[static_cast<size_t>(t->second)].get<T>();
    }
    return T();
  }

private:
  Result begin_argument(const std::string &arg, bool longarg, int position) {
    auto it = positional_arguments_.find(position);
    if (it != positional_arguments_.end()) {
      Result err = end_argument();
      Argument &a = arguments_[static_cast<size_t>(it->second)];
      a.values_.push_back((longarg ? "--" : "-") + arg);
      a.found_ = true;
      return err;
    }
    if (current_ != -1) {
      return Result("error.");
    }
    size_t name_end = find_name_end(arg);
    std::string arg_name = arg.substr(0, name_end);
    if (longarg) {
      int equal_pos = find_equal(arg);
      auto nmf = name_map_.find(arg_name);
      if (nmf == name_map_.end()) {
        return Result("Unknown option '" + arg_name + "'");
      }
      current_ = nmf->second;
      arguments_[static_cast<size_t>(nmf->second)].found_ = true;
      if (equal_pos == 0 ||
          (equal_pos < 0 &&
           arg_name.length() < arg.length())) { // malformed argument
        return Result("not valid argument: " + arg);
      }
      if (equal_pos > 0) {
        std::string arg_value = arg.substr(name_end + 1);
        add_value(arg_value, position);
      }
    } else {
      Result r;
      if (arg_name.length() == 1) {
        return begin_argument(arg, true, position);
      }
      for (char &c : arg_name) {
        r = begin_argument(std::string(1, c), true, position);
        if (r) {
          return r;
        }
        r = end_argument();
        if (r) {
          return r;
        }
      }
    }
    return Result();
  }

  Result add_value(const std::string &value, int location) {
    if (current_ >= 0) {
      Result err;
      Argument &a = arguments_[static_cast<size_t>(current_)];
      if (a.count_ >= 0 && static_cast<int>(a.values_.size()) >= a.count_) {
        err = end_argument();
        if (err) {
          return err;
        }
        goto unnamed;
      }
      a.values_.push_back(value);
      if (a.count_ >= 0 && static_cast<int>(a.values_.size()) >= a.count_) {
        err = end_argument();
        if (err) {
          return err;
        }
      }
      return Result();
    }
  unnamed:
    auto it = positional_arguments_.find(location);
    if (it != positional_arguments_.end()) {
      Argument &a = arguments_[static_cast<size_t>(it->second)];
      a.values_.push_back(value);
      a.found_ = true;
    }
    return Result();
  }

  Result end_argument() {
    if (current_ >= 0) {
      Argument &a = arguments_[static_cast<size_t>(current_)];
      current_ = -1;
      if (static_cast<int>(a.values_.size()) < a.count_) {
        return Result("Too few arguments given for " + a.names_[0]);
      }
      if (a.count_ >= 0) {
        if (static_cast<int>(a.values_.size()) > a.count_) {
          return Result("Too many arguments given for " + a.names_[0]);
        }
      }
    }
    return Result();
  }

  bool help_enabled_{false};
  int current_{-1};
  std::string bin_{};
  std::string desc_{};
  std::vector<Argument> arguments_{};
  std::map<int, int> positional_arguments_{};
  std::map<std::string, int> name_map_{};
};

inline std::ostream &operator<<(std::ostream &os, const ArgParser::Result &r) {
  os << r.what();
  return os;
}
template <> inline std::string ArgParser::Argument::get<std::string>() {
  return join(values_.begin(), values_.end());
}
template <>
inline std::vector<std::string>
ArgParser::Argument::get<std::vector<std::string>>() {
  return values_;
}

} // namespace arg
#endif