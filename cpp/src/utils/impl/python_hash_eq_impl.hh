/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_HASH_EQ_IMPL_HH
#define SKDECIDE_PYTHON_HASH_EQ_IMPL_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <boost/container_hash/hash.hpp>

#include "utils/logging.hh"
#include "utils/python_globals.hh"
#include "utils/python_gil_control.hh"
#include "utils/python_container_proxy.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution>
bool PythonEqual<Texecution>::operator()(const py::object &o1,
                                         const py::object &o2) const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    std::function<bool(const py::object &, const py::object &, bool &)>
        compute_equal =
            [](const py::object &eo1, const py::object &eo2, bool &eq_test) {
              py::object res = eo1.attr("__eq__")(eo2);
              if (!res.is(skdecide::Globals::not_implemented_object())) {
                eq_test = res.template cast<bool>();
                return true;
              } else {
                return false;
              }
            };
    if (py::isinstance<py::array>(o1) && py::isinstance<py::array>(o2)) {
      return py::module::import("numpy")
          .attr("array_equal")(o1, o2)
          .template cast<bool>();
    } else {
      bool eq_test = false;
      if (!py::hasattr(o1, "__eq__") || o1.attr("__eq__").is_none() ||
          !py::hasattr(o2, "__eq__") || o2.attr("__eq__").is_none() ||
          !compute_equal(o1, o2, eq_test)) {
        // Try to equalize using __repr__
        py::object r1 = o1.attr("__repr__")();
        py::object r2 = o2.attr("__repr__")();
        if (!py::hasattr(r1, "__eq__") || r1.attr("__eq__").is_none() ||
            !py::hasattr(r2, "__eq__") || r2.attr("__eq__").is_none() ||
            !compute_equal(r1, r2, eq_test)) {
          // Try to equalize using __str__
          py::object s1 = o1.attr("__str__")();
          py::object s2 = o2.attr("__str__")();
          if (!py::hasattr(s1, "__eq__") || s1.attr("__eq__").is_none() ||
              !py::hasattr(s2, "__eq__") || s2.attr("__eq__").is_none() ||
              !compute_equal(s1, s2, eq_test)) {
            // Desperate case...
            throw std::invalid_argument(
                "SKDECIDE exception: python objects do not provide usable "
                "__eq__ nor equal tests using __repr__ or __str__");
          }
        }
      }
      return eq_test;
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string(
            "SKDECIDE exception when testing equality of python objects: ") +
        e->what());
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

template <typename Texecution>
std::size_t PythonHash<Texecution>::operator()(const py::object &o) const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    std::function<bool(const py::object &, std::size_t &)> compute_hash =
        [](const py::object &ho, std::size_t &hash_val) {
          py::object res = ho.attr("__hash__")();
          if (!res.is(skdecide::Globals::not_implemented_object())) {
            // python __hash__ can return negative integers but c++ expects
            // positive integers only return  (ho.attr("__hash__")().template
            // cast<std::size_t>()) % ((skdecide::Globals::python_sys_maxsize()
            // + 1) * 2);
            hash_val = (res.template cast<long long>()) +
                       skdecide::Globals::python_sys_maxsize();
            return true;
          } else {
            return false;
          }
        };
    // special cases
    if (py::isinstance<py::array>(o)) {
      return PythonContainerProxy<Texecution>(o).hash();
    } else if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
      // we could use PythonContainerProxy<Texecution>(o).hash() but it involves
      // copies and redirections...
      std::size_t seed = 0;
      for (auto e : o) {
        boost::hash_combine(seed,
                            ItemHasher(py::reinterpret_borrow<py::object>(e)));
      }
      return seed;
    } else if (py::isinstance<py::set>(o)) {
      std::size_t seed = 0;
      py::list keys = py::cast<py::list>(skdecide::Globals::sorted()(o));
      for (auto k : keys) {
        boost::hash_combine(seed,
                            ItemHasher(py::reinterpret_borrow<py::object>(k)));
      }
      return seed;
    } else if (py::isinstance<py::dict>(o)) {
      std::size_t seed = 0;
      py::list keys = py::cast<py::list>(skdecide::Globals::sorted()(o));
      for (auto k : keys) {
        boost::hash_combine(seed,
                            ItemHasher(py::reinterpret_borrow<py::object>(k)));
        boost::hash_combine(seed, ItemHasher(o[k]));
      }
      return seed;
    } else { // normal cases
      std::size_t hash_val = 0;
      if (!py::hasattr(o, "__hash__") || o.attr("__hash__").is_none() ||
          !compute_hash(o, hash_val)) {
        // Try to hash using __repr__
        py::object r = o.attr("__repr__")();
        if (!py::hasattr(r, "__hash__") || r.attr("__hash__").is_none() ||
            !compute_hash(r, hash_val)) {
          // Try to hash using __str__
          py::object s = o.attr("__str__")();
          if (!py::hasattr(s, "__hash__") || s.attr("__hash__").is_none() ||
              !compute_hash(s, hash_val)) {
            // Desperate case...
            throw std::invalid_argument(
                "SKDECIDE exception: python object does not provide usable "
                "__hash__ nor hashable __repr__ or __str__");
          }
        }
      }
      return hash_val;
    }
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when hashing python object: ") +
        e->what());
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

template <typename Texecution>
PythonHash<Texecution>::ItemHasher::ItemHasher(const py::object &o)
    : _pyobj(o) {}

template <typename Texecution>
std::size_t PythonHash<Texecution>::ItemHasher::hash() const {
  return PythonHash<Texecution>()(_pyobj);
}

inline std::size_t
hash_value(const PythonHash<SequentialExecution>::ItemHasher &ih) {
  return ih.hash();
}

inline std::size_t
hash_value(const PythonHash<ParallelExecution>::ItemHasher &ih) {
  return ih.hash();
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_HASH_EQ_IMPL_HH
