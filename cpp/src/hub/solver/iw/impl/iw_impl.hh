/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_IW_IMPL_HH
#define SKDECIDE_IW_IMPL_HH

#include <boost/range/irange.hpp>
#include <iostream>

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === DomainStateHash implementation ===

#define SK_IW_DOMAIN_STATE_HASH_TEMPLATE_DECL                                  \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_IW_DOMAIN_STATE_HASH_CLASS DomainStateHash<Tdomain, Tfeature_vector>

SK_IW_DOMAIN_STATE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_IW_DOMAIN_STATE_HASH_CLASS::Key &
SK_IW_DOMAIN_STATE_HASH_CLASS::get_key(const Tnode &n) {
  return n.state;
}

SK_IW_DOMAIN_STATE_HASH_TEMPLATE_DECL
std::size_t
SK_IW_DOMAIN_STATE_HASH_CLASS::Hash::operator()(const Key &k) const {
  return typename Tdomain::State::Hash()(k);
}

SK_IW_DOMAIN_STATE_HASH_TEMPLATE_DECL
bool SK_IW_DOMAIN_STATE_HASH_CLASS::Equal::operator()(const Key &k1,
                                                      const Key &k2) const {
  return typename Tdomain::State::Equal()(k1, k2);
}

// === StateFeatureHash implementation ===

#define SK_IW_STATE_FEATURE_HASH_TEMPLATE_DECL                                 \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_IW_STATE_FEATURE_HASH_CLASS                                         \
  StateFeatureHash<Tdomain, Tfeature_vector>

SK_IW_STATE_FEATURE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_IW_STATE_FEATURE_HASH_CLASS::Key &
SK_IW_STATE_FEATURE_HASH_CLASS::get_key(const Tnode &n) {
  return *n.features;
}

SK_IW_STATE_FEATURE_HASH_TEMPLATE_DECL
std::size_t
SK_IW_STATE_FEATURE_HASH_CLASS::Hash::operator()(const Key &k) const {
  std::size_t seed = 0;
  for (std::size_t i = 0; i < k.size(); i++) {
    boost::hash_combine(seed, k[i]);
  }
  return seed;
}

SK_IW_STATE_FEATURE_HASH_TEMPLATE_DECL
bool SK_IW_STATE_FEATURE_HASH_CLASS::Equal::operator()(const Key &k1,
                                                       const Key &k2) const {
  std::size_t size = k1.size();
  if (size != k2.size()) {
    return false;
  }
  for (std::size_t i = 0; i < size; i++) {
    if (!(k1[i] == k2[i])) {
      return false;
    }
  }
  return true;
}

// === IWSolver implementation ===

#define SK_IW_SOLVER_TEMPLATE_DECL                                             \
  template <typename Tdomain, typename Tfeature_vector,                        \
            template <typename...> class Thashing_policy,                      \
            typename Texecution_policy>

#define SK_IW_SOLVER_CLASS                                                     \
  IWSolver<Tdomain, Tfeature_vector, Thashing_policy, Texecution_policy>

SK_IW_SOLVER_TEMPLATE_DECL
SK_IW_SOLVER_CLASS::IWSolver(
    Domain &domain,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &d, const State &s)> &state_features,
    const std::function<bool(const double &, const std::size_t &,
                             const std::size_t &, const double &,
                             const std::size_t &, const std::size_t &)>
        &node_ordering,
    std::size_t time_budget, // time budget to continue searching for better
                             // plans after a goal has been reached
    bool debug_logs)
    : _domain(domain), _state_features(state_features),
      _time_budget(time_budget), _debug_logs(debug_logs) {

  if (!node_ordering) {
    _node_ordering = [](const double &a_gscore, const std::size_t &a_novelty,
                        const std::size_t &a_depth, const double &b_gscore,
                        const std::size_t &b_novelty,
                        const std::size_t &b_depth) -> bool {
      return a_gscore > b_gscore;
    };
  } else {
    _node_ordering = node_ordering;
  }

  if (debug_logs) {
    Logger::check_level(logging::debug, "algorithm IW");
  }
}

SK_IW_SOLVER_TEMPLATE_DECL
void SK_IW_SOLVER_CLASS::clear() { _graph.clear(); }

SK_IW_SOLVER_TEMPLATE_DECL
void SK_IW_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " IW solver from state " + s.print());
    auto start_time = std::chrono::high_resolution_clock::now();
    std::size_t nb_of_binary_features = _state_features(_domain, s)->size();
    bool found_goal = false;
    _intermediate_scores.clear();

    for (std::size_t w = 1; w <= nb_of_binary_features; w++) {
      std::pair<bool, bool> res =
          WidthSolver(_domain, _state_features, _node_ordering, w, _graph,
                      _time_budget, _intermediate_scores, _debug_logs)
              .solve(s, start_time, found_goal);
      if (res.first) { // solution found with width w
        auto now_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_time - start_time)
                            .count();

        if (static_cast<std::size_t>(duration) < _time_budget) {
          if (_debug_logs)
            Logger::debug("Remaining time budget, trying to improve the "
                          "solution by augmenting the search width");
        } else {
          auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now_time - start_time)
                              .count();
          Logger::info("IW finished to solve from state " + s.print() + " in " +
                       StringConverter::from((double)duration / (double)1e9) +
                       " seconds.");
          return;
        }
      } else if (found_goal) {
        auto now_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now_time - start_time)
                            .count();
        Logger::info("IW finished to solve from state " + s.print() + " in " +
                     StringConverter::from((double)duration / (double)1e9) +
                     " seconds.");
        return;
      } else if (!res.second) { // no states pruned => problem is unsolvable
        break;
      }
    }

    Logger::info("IW could not find a solution from state " + s.print());
  } catch (const std::exception &e) {
    Logger::error("IW failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_IW_SOLVER_TEMPLATE_DECL
bool SK_IW_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if ((si == _graph.end()) || (si->best_action == nullptr) ||
      (si->solved == false)) {
    return false;
  } else {
    return true;
  }
}

SK_IW_SOLVER_TEMPLATE_DECL
const typename SK_IW_SOLVER_CLASS::Action &
SK_IW_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return *(si->best_action);
}

SK_IW_SOLVER_TEMPLATE_DECL
const double &SK_IW_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->fscore;
}

SK_IW_SOLVER_TEMPLATE_DECL
std::size_t SK_IW_SOLVER_CLASS::get_nb_of_explored_states() const {
  return _graph.size();
}

SK_IW_SOLVER_TEMPLATE_DECL
std::size_t SK_IW_SOLVER_CLASS::get_nb_of_pruned_states() const {
  std::size_t cnt = 0;
  for (const auto &n : _graph) {
    if (n.pruned) {
      cnt++;
    }
  }
  return cnt;
}

SK_IW_SOLVER_TEMPLATE_DECL
const std::list<std::tuple<std::size_t, std::size_t, double>> &
SK_IW_SOLVER_CLASS::get_intermediate_scores() const {
  return _intermediate_scores;
}

// === IWSolver::Node implementation ===

SK_IW_SOLVER_TEMPLATE_DECL
SK_IW_SOLVER_CLASS::Node::Node(
    const State &s, Domain &d,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &d, const State &s)> &state_features)
    : state(s), gscore(std::numeric_limits<double>::infinity()),
      fscore(std::numeric_limits<double>::infinity()),
      novelty(std::numeric_limits<std::size_t>::max()),
      depth(std::numeric_limits<std::size_t>::max()), best_action(nullptr),
      pruned(false), solved(false) {
  features = state_features(d, s);
}

SK_IW_SOLVER_TEMPLATE_DECL
const typename SK_IW_SOLVER_CLASS::HashingPolicy::Key &
SK_IW_SOLVER_CLASS::Node::Key::operator()(const Node &n) const {
  return HashingPolicy::get_key(n);
}

// === IWSolver::WidthSolver implementation ===

SK_IW_SOLVER_TEMPLATE_DECL
SK_IW_SOLVER_CLASS::WidthSolver::WidthSolver(
    Domain &domain,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &d, const State &s)> &state_features,
    const std::function<bool(const double &, const std::size_t &,
                             const std::size_t &, const double &,
                             const std::size_t &, const std::size_t &)>
        &node_ordering,
    std::size_t width, Graph &graph, std::size_t time_budget,
    std::list<std::tuple<std::size_t, std::size_t, double>>
        &intermediate_scores,
    bool debug_logs)
    : _domain(domain), _state_features(state_features),
      _node_ordering(node_ordering), _width(width), _graph(graph),
      _time_budget(time_budget), _intermediate_scores(intermediate_scores),
      _debug_logs(debug_logs) {}

SK_IW_SOLVER_TEMPLATE_DECL
std::pair<bool, bool> SK_IW_SOLVER_CLASS::WidthSolver::solve(
    const State &s,
    const std::chrono::time_point<std::chrono::high_resolution_clock>
        &start_time,
    bool &found_goal) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() + " IW(" +
                 StringConverter::from(_width) + ") solver from state " +
                 s.print());
    auto local_start_time = std::chrono::high_resolution_clock::now();

    // Create the root node containing the given state s
    auto si = _graph.emplace(Node(s, _domain, _state_features));
    if (si.first->solved ||
        _domain.is_goal(s)) { // problem already solved from this state (was
                              // present in _graph and already solved)
      return std::make_pair(true, false);
    } else if (_domain.is_terminal(s)) { // dead-end state
      return std::make_pair(false, false);
    }
    Node &root_node = const_cast<Node &>(*(
        si.first)); // we won't change the real key (Node::state) so we are safe
    root_node.depth = 0;
    root_node.gscore = 0;
    bool states_pruned = false;

    auto node_ordering = [this](const auto &a, const auto &b) -> bool {
      return _node_ordering(a->gscore, a->novelty, a->depth, b->gscore,
                            b->novelty, b->depth);
    };

    // Priority queue used to sort non-goal unsolved tip nodes by increasing
    // cost-to-go values (so-called OPEN container)
    std::priority_queue<Node *, std::vector<Node *>, decltype(node_ordering)>
        open_queue(node_ordering);
    open_queue.push(&root_node);

    // Set of states that have already been explored
    std::unordered_set<Node *> closed_set;

    // Vector of sets of state feature tuples generated so far, for each w <=
    // _width
    TupleVector feature_tuples(_width);
    novelty(feature_tuples,
            root_node); // initialize feature_tuples with the root node's bits

    while (!open_queue.empty()) {
      auto best_tip_node = open_queue.top();
      open_queue.pop();

      // Check that the best tip node has not already been closed before
      // (since this implementation's open_queue does not check for element
      // uniqueness, it can contain many copies of the same node pointer that
      // could have been closed earlier)
      if (closed_set.find(best_tip_node) !=
          closed_set
              .end()) { // this implementation's open_queue can contain several
        continue;
      }

      if (_debug_logs)
        Logger::debug(
            "Current best tip node: " + best_tip_node->state.print() +
            ", gscore=" + StringConverter::from(best_tip_node->gscore) +
            ", novelty=" + StringConverter::from(best_tip_node->novelty) +
            ", depth=" + StringConverter::from(best_tip_node->depth));

      closed_set.insert(best_tip_node);

      if (_domain.is_goal(best_tip_node->state) || best_tip_node->solved) {
        auto current_node = best_tip_node;
        double tentative_fscore = 0;

        while (current_node != &root_node) {
          Node *parent_node = std::get<0>(current_node->best_parent);
          tentative_fscore += std::get<2>(current_node->best_parent);
          current_node = parent_node;
        }

        if (!found_goal || root_node.fscore > tentative_fscore) {
          current_node = best_tip_node;
          if (!(best_tip_node->solved)) {
            current_node->fscore = 0;
          } // goal state
          best_tip_node->solved = true;

          while (current_node != &root_node) {
            Node *parent_node = std::get<0>(current_node->best_parent);
            parent_node->best_action = &std::get<1>(current_node->best_parent);
            parent_node->fscore =
                std::get<2>(current_node->best_parent) + current_node->fscore;
            parent_node->solved = true;
            current_node = parent_node;
          }
        }

        Logger::info("Found a goal state: " + best_tip_node->state.print() +
                     " (cost=" + StringConverter::from(tentative_fscore) +
                     "; best=" + StringConverter::from(root_node.fscore) + ")");
        found_goal = true;
        auto now_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now_time - start_time)
                            .count();
        _intermediate_scores.push_back(std::make_tuple(
            static_cast<std::size_t>(duration), _width, root_node.fscore));

        if (static_cast<std::size_t>(duration) < _time_budget) {
          if (_debug_logs)
            Logger::debug("Remaining time budget, continuing searching for "
                          "better plans with current width...");
          continue;
        } else {
          auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now_time - local_start_time)
                              .count();
          Logger::info("IW(" + StringConverter::from(_width) +
                       ") finished to solve from state " + s.print() + " in " +
                       StringConverter::from((double)duration / (double)1e9) +
                       " seconds.");
          return std::make_pair(true, states_pruned);
        }
      }

      if (_domain.is_terminal(best_tip_node->state)) { // dead-end state
        if (_debug_logs)
          Logger::debug("Found a dead-end state: " +
                        best_tip_node->state.print());
        continue;
      }

      // Expand best tip node
      auto applicable_actions =
          _domain.get_applicable_actions(best_tip_node->state).get_elements();
      std::for_each(
          ExecutionPolicy::policy, applicable_actions.begin(),
          applicable_actions.end(),
          [this, &best_tip_node, &open_queue, &feature_tuples,
           &states_pruned](auto a) {
            if (_debug_logs)
              Logger::debug("Current expanded action: " + a.print() +
                            ExecutionPolicy::print_thread());
            auto next_state = _domain.get_next_state(best_tip_node->state, a);
            std::pair<typename Graph::iterator, bool> i;
            _execution_policy.protect([this, &i, &next_state] {
              i = _graph.emplace(Node(next_state, _domain, _state_features));
            });
            Node &neighbor = const_cast<Node &>(
                *(i.first)); // we won't change the real key (StateNode::state)
                             // so we are safe
            if (_debug_logs)
              Logger::debug("Exploring next state (among " +
                            StringConverter::from(_graph.size()) + ")" +
                            ExecutionPolicy::print_thread());

            double transition_cost =
                _domain
                    .get_transition_value(best_tip_node->state, a,
                                          neighbor.state)
                    .cost();
            double tentative_gscore = best_tip_node->gscore + transition_cost;
            std::size_t tentative_depth = best_tip_node->depth + 1;

            if ((i.second) || (tentative_gscore < neighbor.gscore)) {
              if (_debug_logs)
                Logger::debug("New gscore: " +
                              StringConverter::from(best_tip_node->gscore) +
                              "+" + StringConverter::from(transition_cost) +
                              "=" + StringConverter::from(tentative_gscore) +
                              ExecutionPolicy::print_thread());
              neighbor.gscore = tentative_gscore;
              neighbor.best_parent =
                  std::make_tuple(best_tip_node, a, transition_cost);
            }

            if ((i.second) || (tentative_depth < neighbor.depth)) {
              if (_debug_logs)
                Logger::debug("New depth: " +
                              StringConverter::from(best_tip_node->depth) +
                              "+" + StringConverter::from(1) + "=" +
                              StringConverter::from(tentative_depth) +
                              ExecutionPolicy::print_thread());
              neighbor.depth = tentative_depth;
            }

            _execution_policy.protect([this, &feature_tuples, &open_queue,
                                       &neighbor, &states_pruned] {
              if (this->novelty(feature_tuples, neighbor) > _width) {
                if (_debug_logs)
                  Logger::debug("Pruning state " + neighbor.state.print() +
                                ExecutionPolicy::print_thread());
                states_pruned = true;
                neighbor.pruned = true;
              } else {
                if (_debug_logs)
                  Logger::debug("Adding state to open queue (among " +
                                StringConverter::from(open_queue.size()) + ")" +
                                ExecutionPolicy::print_thread());
                open_queue.push(&neighbor);
              }
            });
          });
    }

    if (found_goal) {
      auto now_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now_time - local_start_time)
                          .count();
      Logger::info("IW(" + StringConverter::from(_width) +
                   ") finished to solve from state " + s.print() + " in " +
                   StringConverter::from((double)duration / (double)1e9) +
                   " seconds.");
      return std::make_pair(true, states_pruned);
    } else {
      Logger::info("IW(" + StringConverter::from(_width) +
                   ") could not find a solution from state " + s.print());
      return std::make_pair(false, states_pruned);
    }
  } catch (const std::exception &e) {
    Logger::error("IW(" + StringConverter::from(_width) +
                  ") failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_IW_SOLVER_TEMPLATE_DECL
std::size_t
SK_IW_SOLVER_CLASS::WidthSolver::novelty(TupleVector &feature_tuples,
                                         Node &n) const {
  // feature_tuples is a set of state variable combinations of size _width
  std::size_t nov = n.features->size() + 1;
  const FeatureVector &state_features = *n.features;

  for (std::size_t k = 1; k <= std::min(_width, state_features.size()); k++) {
    // we must recompute combinations from previous width values just in case
    // this state would be visited for the first time across width iterations
    generate_tuples(
        k, state_features.size(),
        [&state_features, &feature_tuples, &k, &nov](TupleType &cv) {
          for (auto &e : cv) {
            e.second = state_features[e.first];
          }
          if (feature_tuples[k - 1].insert(cv).second) {
            nov = std::min(nov, k);
          }
        });
  }
  if (_debug_logs)
    Logger::debug("Novelty: " + StringConverter::from(nov));
  n.novelty = nov;
  return nov;
}

SK_IW_SOLVER_TEMPLATE_DECL
void SK_IW_SOLVER_CLASS::WidthSolver::generate_tuples(
    const std::size_t &k, const std::size_t &n,
    const std::function<void(TupleType &)> &f) const {
  TupleType cv(k); // one combination (the first one)
  for (std::size_t i = 0; i < k; i++) {
    cv[i].first = i;
  }
  f(cv);
  bool more_combinations = true;
  while (more_combinations) {
    more_combinations = false;
    // find the rightmost element that has not yet reached its highest possible
    // value
    for (std::size_t i = k; i > 0; i--) {
      if (cv[i - 1].first < n - k + i - 1) {
        // once finding this element, we increment it by 1,
        // and assign the lowest valid value to all subsequent elements
        cv[i - 1].first++;
        for (std::size_t j = i; j < k; j++) {
          cv[j].first = cv[j - 1].first + 1;
        }
        f(cv);
        more_combinations = true;
        break;
      }
    }
  }
}

} // namespace skdecide

#endif // SKDECIDE_IW_IMPL_HH
