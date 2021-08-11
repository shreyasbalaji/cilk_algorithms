#ifndef CILKSTL_ALGORITHM_H
#define CILKSTL_ALGORITHM_H

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_min.h>
#include <cilk/reducer_opadd.h>

#include <atomic>
#include <cstdlib>
#include <iterator>

namespace cilkstl {
namespace __parallel {

/**
 * Rotate implementation using an uninitialized buffer. Rotates the array by moving the data from the larger segment to
 * the temporary buffer, rotating the data from the smaller segment to the larger segment, and then rotating the data
 * from the larger segment to the smaller segment. PSTL implements something similar except they distinguish between
 * uninitialized and initialized moves
 */
template <class _RandomAccessIterator>
_RandomAccessIterator rotate(_RandomAccessIterator first, _RandomAccessIterator middle, _RandomAccessIterator last) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  diff_t a = middle - first;
  diff_t b = last - middle;
  diff_t c = last - first;

  // Picks the larger segment, copies into buffer
  // Rotates smaller segment into proper position
  // Loads larger segment from buffer and rotates into proper position
  if (a <= c / 2) {
    value_t *buffer = new value_t[b];
    cilk_for(diff_t k = 0; k < b; ++k) { *(buffer + k) = std::move(*(middle + k)); }
    cilk_for(diff_t k = 0; k < a; ++k) { *(first + b + k) = std::move(*(first + k)); }
    cilk_for(diff_t k = 0; k < b; ++k) { *(first + k) = std::move(*(buffer + k)); }
    delete[] buffer;
  } else {
    value_t *buffer = new value_t[a];
    cilk_for(diff_t k = 0; k < a; ++k) { *(buffer + k) = std::move(*(first + k)); }
    cilk_for(diff_t k = 0; k < b; ++k) { *(first + k) = std::move(*(middle + k)); }
    cilk_for(diff_t k = 0; k < a; ++k) { *(first + b + k) = std::move(*(buffer + k)); }
    delete[] buffer;
  }

  return first + b;
}

/**
 * Rotate implementation using no additional memory. Reverses ranges [first, middle) and [middle, last). Then reverses
 * the whole array [first, last).
 */
template <class _RandomAccessIterator>
_RandomAccessIterator rotate_inplace(_RandomAccessIterator first, _RandomAccessIterator middle,
                                     _RandomAccessIterator last) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  // Defines a function that parallel reverses range [start, end) inplace
  auto _reverse_inplace = [&](_RandomAccessIterator start, _RandomAccessIterator end) {
    cilk_for(diff_t k = 0; k < (end - start) / 2; ++k) {
      value_t tmp = std::move(*(start + k));
      *(start + k) = std::move(*(end - k - 1));
      *(end - k - 1) = std::move(tmp);
    }
  };

  cilk_spawn _reverse_inplace(first, middle);
  _reverse_inplace(middle, last);
  cilk_sync;
  _reverse_inplace(first, last);

  return first + (last - middle);
}

/**
 * Implements spec from std::transform by applying `transform_func` in a cilk_for loop.
 */
template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _UnaryOperation>
_RandomAccessIterator2 transform(_RandomAccessIterator1 first, _RandomAccessIterator1 last,
                                 _RandomAccessIterator2 d_first, _UnaryOperation transform_func) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator1>::difference_type diff1_t;

  diff1_t range_width = last - first;
  cilk_for(diff1_t diff = 0; diff < range_width; ++diff) { *(d_first + diff) = transform_func(*(first + diff)); }
  return *(d_first + range_width);
}

/**
 * Implements spec from std::max_element by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator>
_RandomAccessIterator max_element(_RandomAccessIterator first, _RandomAccessIterator last) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  cilk::reducer<cilk::op_max_index<diff_t, value_t>> max_rd;
  diff_t range_width = last - first;
  cilk_for(diff_t diff = 0; diff < range_width; ++diff) { max_rd->calc_max(diff, *(first + diff)); }
  return first + (max_rd.get_value().first);
}

/**
 * Implements spec from std::max_element by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator, class _Compare>
_RandomAccessIterator max_element(_RandomAccessIterator first, _RandomAccessIterator last, _Compare comp) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  cilk::reducer<cilk::op_max_index<diff_t, value_t, _Compare>> max_rd(comp);
  diff_t range_width = last - first;
  cilk_for(diff_t diff = 0; diff < range_width; ++diff) { max_rd->calc_max(diff, *(first + diff)); }
  return first + (max_rd.get_value().first);
}

/**
 * Implements spec from std::min_element by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator>
_RandomAccessIterator min_element(_RandomAccessIterator first, _RandomAccessIterator last) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  cilk::reducer<cilk::op_min_index<diff_t, value_t>> min_rd;
  diff_t range_width = last - first;
  cilk_for(diff_t diff = 0; diff < range_width; ++diff) { min_rd->calc_min(diff, *(first + diff)); }
  return first + (min_rd.get_value().first);
}

/**
 * Implements spec from std::min_element by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator, class _Compare>
_RandomAccessIterator min_element(_RandomAccessIterator first, _RandomAccessIterator last, _Compare comp) {
  if (first >= last)
    return first;

  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;

  cilk::reducer<cilk::op_min_index<diff_t, value_t, _Compare>> min_rd(comp);
  diff_t range_width = last - first;
  cilk_for(diff_t diff = 0; diff < range_width; ++diff) { min_rd->calc_min(diff, *(first + diff)); }
  return first + (min_rd.get_value().first);
}

/**
 * Implements spec from std::count by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator, class _Type>
typename std::iterator_traits<_RandomAccessIterator>::difference_type
count(_RandomAccessIterator first, _RandomAccessIterator last, const _Type &value) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  cilk::reducer<cilk::op_add<diff_t>> count_rd;

  cilk_for(diff_t k = 0; k < range_width; ++k) {
    if (*(first + k) == value)
      *count_rd += 1;
  }

  return count_rd.get_value();
}

/**
 * Implements spec from std::count_if by iterating through the array in a cilk_for loop and tracking the result with
 * a cilk reducer.
 */
template <class _RandomAccessIterator, class _PredicateFunc>
typename std::iterator_traits<_RandomAccessIterator>::difference_type
count_if(_RandomAccessIterator first, _RandomAccessIterator last, _PredicateFunc predicate) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  cilk::reducer<cilk::op_add<diff_t>> count_rd;

  cilk_for(diff_t k = 0; k < range_width; ++k) {
    if (predicate(*(first + k)))
      *count_rd += 1;
  }

  return count_rd.get_value();
}

// Grain size that determines cutoff to switch to serial code for parallel code that splits the range in half and
// recurses into each half in parallel
constexpr int BINARY_GRAIN_SIZE = 2000;

/**
 * Implements spec from std::is_sorted by splitting the array in half and recursively solving each half in parallel.
 */
template <class _RandomAccessIterator, class _Compare>
bool is_sorted(_RandomAccessIterator first, _RandomAccessIterator last, _Compare comp) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  if (range_width < 2)
    return true;

  // default to serial code if problem size is too small
  _RandomAccessIterator second_last = last - 1;
  if (range_width < BINARY_GRAIN_SIZE) {
    for (auto it = first; it < second_last; ++it) {
      if (comp(*(it + 1), *it))
        return false;
    }
    return true;
  }

  _RandomAccessIterator middle = first + (range_width / 2);

  // handle edge case where middle - 1 is in left spawn but middle is in right spawn
  if (comp(*middle, *(middle - 1)))
    return false;

  // recursively spawn left and right halves
  bool first_sorted = cilk_spawn is_sorted(first, middle, comp);
  bool second_sorted = is_sorted(middle, last, comp);
  cilk_sync;

  // return result
  return (first_sorted && second_sorted);
}

/**
 * Implements spec from std::find by splitting the array in half and recursively solving each half in parallel.
 */
template <class _RandomAccessIterator, class T>
_RandomAccessIterator find(_RandomAccessIterator first, _RandomAccessIterator last, const T &value) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;

  // default to serial if problem size is too small
  if (range_width < BINARY_GRAIN_SIZE) {
    return std::find(first, last, value);
  }

  _RandomAccessIterator middle = first + (range_width / 2);

  // spawn each half in parallel
  _RandomAccessIterator left_find = cilk_spawn cilkstl::__parallel::find(first, middle, value);
  _RandomAccessIterator right_find = cilkstl::__parallel::find(middle, last, value);
  cilk_sync;

  // combine results
  return (left_find != middle) ? left_find : right_find;
}

// Grain size for parallel find2 function
constexpr int FIND2_GRAIN_SIZE = 2400;

/**
 * Helper function for find2 that contains the logic to split the problem into halves and recurse in parallel, assigning
 * the result to an atomic variable. Each recursive call is prefaced by a check to the atomic variable to avoid
 * unnecessary work if a better index has already been found.
 */
template <class _RandomAccessIterator, class T>
void __find2(_RandomAccessIterator begin, typename std::iterator_traits<_RandomAccessIterator>::difference_type start,
             typename std::iterator_traits<_RandomAccessIterator>::difference_type end, const T &value,
             std::atomic<typename std::iterator_traits<_RandomAccessIterator>::difference_type> &idx) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = end - start;

  // Only do work if range represented by this recursive call includes values less than
  // the current lowest found index
  if (start < idx) {
    if (range_width < FIND2_GRAIN_SIZE) {
      // for small arrays, run find in serial and update the atomic variable `idx` containing the result as needed
      _RandomAccessIterator result = std::find(begin + start, begin + end, value);
      diff_t r = result - begin;
      if (r < end) {
        for (diff_t z = idx; r < z; z = idx) {
          idx.compare_exchange_weak(z, r);
        }
      }
    } else {
      // recurse into two array halves
      diff_t middle = start + range_width / 2;
      cilk_spawn cilkstl::__parallel::__find2(begin, start, middle, value, idx);
      cilkstl::__parallel::__find2(begin, middle, end, value, idx);
    }
  }

  return;
}

/**
 * Implements spec from std::find by splitting the array in half and recursively solving each half in parallel, using
 *  an atomic variable to keep track of the result.
 */
template <class _RandomAccessIterator, class T>
_RandomAccessIterator find2(_RandomAccessIterator first, _RandomAccessIterator last, const T &value) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  if (range_width <= 2 * FIND2_GRAIN_SIZE) {
    return std::find(first, last, value);
  }

  std::atomic<diff_t> idx(range_width); // stores lowest found index that matches value
  ::cilkstl::__parallel::__find2(first, first - first, last - first, value, idx);
  return first + idx;
}

} // namespace __parallel
}; // namespace cilkstl

#endif
