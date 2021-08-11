#include "../cilkstl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

class TypedDataSpace {
public:
  static std::int64_t id_count;

  std::int64_t id;
  std::int64_t type;
  std::int64_t data[12];

  friend bool operator<(const TypedDataSpace &lhs, const TypedDataSpace &rhs) { return (lhs.type < rhs.type); }

  inline bool operator==(const TypedDataSpace &rhs) { return (id == rhs.id); }

  TypedDataSpace() { id = id_count++; }

  TypedDataSpace(const TypedDataSpace &rhs) : type(rhs.type) {
    id = id_count++;
    copy_data(rhs);
  }

  TypedDataSpace(const TypedDataSpace &&rhs) : id(rhs.id), type(rhs.type) { copy_data(rhs); }

  TypedDataSpace &operator=(const TypedDataSpace &rhs) {
    if (this == &rhs)
      return *this;
    id = id_count++;
    type = rhs.type;
    copy_data(rhs);
    return *this;
  }

  TypedDataSpace &operator=(const TypedDataSpace &&rhs) {
    id = rhs.id;
    type = rhs.type;
    copy_data(rhs);
    return *this;
  }

  inline friend std::ostream &operator<<(std::ostream &out, const TypedDataSpace &rhs) {
    out << "[id=" << rhs.id << ", type=" << rhs.type << ", data=(" << rhs.data[0] << " " << rhs.data[1] << "...)]";
    return out;
  }

private:
  inline void copy_data(const TypedDataSpace &rhs) {
    for (int i = 0; i < 12; ++i)
      data[i] = rhs.data[i];
  }
};

static std::vector<double> random_vector(size_t size) {
  static std::default_random_engine engine;
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);

  std::vector<double> result(size);
  for (int i = 0; i < size; ++i)
    result[i] = distribution(engine);

  return result;
}

static std::vector<TypedDataSpace> random_typed_vector(size_t size) {
  static std::default_random_engine engine;
  static std::uniform_int_distribution<std::int64_t> type_distribution(0, 10);
  static std::uniform_int_distribution<std::int64_t> data_distribution(0, 100000);

  std::vector<TypedDataSpace> result;
  for (int i = 0; i < size; ++i) {
    result.push_back(TypedDataSpace());
    result[i].type = type_distribution(engine);
    for (int j = 0; j < 12; ++j) {
      result[i].data[j] = data_distribution(engine);
    }
  }

  return result;
}

// TESTS

constexpr int ROTATE_TEST_ARRAY_SIZE = 500000;
constexpr int ROTATE_TEST_REPEATS = 30;

int test_rotate_element() {
  std::vector<double> random_vectors[ROTATE_TEST_REPEATS];
  for (int i = 0; i < ROTATE_TEST_REPEATS; ++i)
    random_vectors[i] = random_vector(ROTATE_TEST_ARRAY_SIZE);

  std::vector<double> cutoff_ratios = random_vector(ROTATE_TEST_REPEATS);

  for (int i = 0; i < ROTATE_TEST_REPEATS; ++i) {
    std::vector<double> random_copy = random_vectors[i];
    std::vector<double>::iterator middle =
        random_vectors[i].begin() + ((int)(cutoff_ratios[i] * random_vectors[i].size()));
    std::vector<double>::iterator middle_copy = random_copy.begin() + ((int)(cutoff_ratios[i] * random_copy.size()));

    cilkstl::__parallel::rotate(random_vectors[i].begin(), middle, random_vectors[i].end());
    std::rotate(random_copy.begin(), middle_copy, random_copy.end());

    for (int j = 0; j < random_vectors[i].size(); ++j) {
      if (random_vectors[i][j] != random_copy[j]) {
        std::cout << "FAIL: test_rotate_element" << std::endl;
        return 1;
      }
    }
  }
  std::cout << "SUCCESS: test_rotate_element" << std::endl;
  return 0;
}

constexpr int MIN_TEST_ARRAY_SIZE = 500000;
constexpr int MIN_TEST_REPEATS = 30;

int test_min_element() {
  std::vector<double> random_vectors[MIN_TEST_REPEATS];
  for (int i = 0; i < MIN_TEST_REPEATS; ++i)
    random_vectors[i] = random_vector(MIN_TEST_ARRAY_SIZE);

  std::vector<double> base_results(MIN_TEST_REPEATS);
  std::vector<double> cilkstl_results(MIN_TEST_REPEATS);

  for (int i = 0; i < MIN_TEST_REPEATS; ++i) {
    base_results[i] = *std::min_element(random_vectors[i].begin(), random_vectors[i].end(), std::less<double>());
  }

  for (int i = 0; i < MIN_TEST_REPEATS; ++i) {
    cilkstl_results[i] =
        *cilkstl::__parallel::min_element(random_vectors[i].begin(), random_vectors[i].end(), std::less<double>());
  }

  for (int i = 0; i < MIN_TEST_REPEATS; ++i) {
    if (std::abs(cilkstl_results[i] - base_results[i]) > 1e-9) {
      std::cout << "FAIL: test_min_element" << std::endl;
      return 1;
    }
  }

  std::cout << "SUCCESS: test_min_element" << std::endl;
  return 0;
}

int test_find() {
  std::vector<double> tmp_doubles = random_vector(20000);
  std::vector<int> v;
  for (int i = 0; i < tmp_doubles.size(); ++i)
    v.push_back((int)(tmp_doubles[i] * 9000));
  for (int i = 1; i < 9040; i += 20) {
    auto base_result = std::find(v.begin(), v.end(), i);
    auto cilkstl_result = cilkstl::__parallel::find(v.begin(), v.end(), i);
    if (base_result != cilkstl_result) {
      std::cout << "FAIL: test_find" << std::endl;
      return 1;
    }
  }

  std::cout << "SUCCESS: test_find" << std::endl;
  return 0;
}

int test_find2() {
  std::vector<double> tmp_doubles = random_vector(20000);
  std::vector<int> v;
  for (int i = 0; i < tmp_doubles.size(); ++i)
    v.push_back((int)(tmp_doubles[i] * 9000));
  for (int i = 1; i < 9040; i += 20) {
    auto base_result = std::find(v.begin(), v.end(), i);
    auto cilkstl_result = cilkstl::__parallel::find2(v.begin(), v.end(), i);
    if (base_result != cilkstl_result) {
      std::cout << "FAIL: test_find2" << std::endl;
      return 1;
    }
  }

  std::cout << "SUCCESS: test_find2" << std::endl;
  return 0;
}

constexpr int SORT_ARRAY_SIZE = 100000;
constexpr int SORT_REPEATS = 20;

int test_stable_sort_correctness1() {
  std::vector<double> random_vectors[SORT_REPEATS];
  for (int i = 0; i < SORT_REPEATS; ++i)
    random_vectors[i] = random_vector(SORT_ARRAY_SIZE);

  for (int i = 0; i < SORT_REPEATS; ++i) {
    std::vector<double> random_copy = random_vectors[i];

    cilkstl::__parallel::__sort::stable_sort(random_vectors[i].begin(), random_vectors[i].end(), std::less<double>());
    std::stable_sort(random_copy.begin(), random_copy.end(), std::less<double>());

    for (int j = 0; j < random_vectors[i].size(); ++j) {
      if (random_vectors[i][j] != random_copy[j]) {
        std::cout << "FAIL: test_stable_sort_correctness1" << std::endl;
        return 1;
      }
    }
  }
  std::cout << "SUCCESS: test_stable_sort_correctness1" << std::endl;
  return 0;
}

int test_stable_sort_correctness2() {
  std::vector<TypedDataSpace> random_vectors[SORT_REPEATS];
  for (int i = 0; i < SORT_REPEATS; ++i)
    random_vectors[i] = random_typed_vector(SORT_ARRAY_SIZE);

  for (int i = 0; i < SORT_REPEATS; ++i) {
    std::vector<TypedDataSpace> random_copy;
    random_copy.reserve(SORT_ARRAY_SIZE);
    for (int j = 0; j < SORT_ARRAY_SIZE; ++j)
      random_copy.push_back(std::move(random_vectors[i][j]));

    std::stable_sort(random_copy.begin(), random_copy.end(), std::less<TypedDataSpace>{});
    cilkstl::__parallel::__sort::stable_sort(random_vectors[i].begin(), random_vectors[i].end(),
                                             std::less<TypedDataSpace>{});

    for (int j = 0; j < random_vectors[i].size(); ++j) {
      if (random_vectors[i][j].id != random_copy[j].id) {
        std::cout << "FAIL: test_stable_sort_correctness2" << std::endl;
        return 1;
      }
    }
  }
  std::cout << "SUCCESS: test_stable_sort_correctness2" << std::endl;
  return 0;
}
