(window.webpackJsonp=window.webpackJsonp||[]).push([[55],{571:function(t,e,a){"use strict";a.r(e);var r=a(38),s=Object(r.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"builders-solver-policy"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#builders-solver-policy"}},[t._v("#")]),t._v(" builders.solver.policy")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),a("skdecide-summary")],1),t._v(" "),a("h2",{attrs:{id:"policies"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#policies"}},[t._v("#")]),t._v(" Policies")]),t._v(" "),a("p",[t._v("A solver must inherit this class if it computes a stochastic policy as part of the solving process.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for"}},[t._v("#")]),t._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action"}},[t._v("#")]),t._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-2"}},[t._v("#")]),t._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-3"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-2"}},[t._v("#")]),t._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-4"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-4"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")]),t._v(" "),a("h2",{attrs:{id:"uncertainpolicies"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#uncertainpolicies"}},[t._v("#")]),t._v(" UncertainPolicies")]),t._v(" "),a("p",[t._v("A solver must inherit this class if it computes a stochastic policy (providing next action distribution\nexplicitly) as part of the solving process.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action-distribution"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution"}},[t._v("#")]),t._v(" get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),t._v(" "),a("p",[t._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-5"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-5"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The probabilistic distribution of next action.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-3"}},[t._v("#")]),t._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-6"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-6"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-3"}},[t._v("#")]),t._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-7"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-7"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-7"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-2"}},[t._v("#")]),t._v(" _get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),t._v(" "),a("p",[t._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-8"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-8"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-8"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The probabilistic distribution of next action.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-4"}},[t._v("#")]),t._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-9"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-9"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-9"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-4"}},[t._v("#")]),t._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-10"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-10"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-10"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")]),t._v(" "),a("h2",{attrs:{id:"deterministicpolicies"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#deterministicpolicies"}},[t._v("#")]),t._v(" DeterministicPolicies")]),t._v(" "),a("p",[t._v("A solver must inherit this class if it computes a deterministic policy as part of the solving process.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action"}},[t._v("#")]),t._v(" get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Get the next deterministic action (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-11"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which next action is requested.")])]),t._v(" "),a("h4",{attrs:{id:"returns-11"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-11"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The next deterministic action.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-3"}},[t._v("#")]),t._v(" get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),t._v(" "),a("p",[t._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-12"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-12"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-12"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The probabilistic distribution of next action.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-5"}},[t._v("#")]),t._v(" is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters-13"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-13"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-13"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-13"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action-5"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-5"}},[t._v("#")]),t._v(" sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-14"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-14"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-14"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-14"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-2"}},[t._v("#")]),t._v(" _get_next_action "),a("Badge",{attrs:{text:"DeterministicPolicies",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Get the next deterministic action (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-15"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-15"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which next action is requested.")])]),t._v(" "),a("h4",{attrs:{id:"returns-15"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-15"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The next deterministic action.")]),t._v(" "),a("h3",{attrs:{id:"get-next-action-distribution-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-next-action-distribution-4"}},[t._v("#")]),t._v(" _get_next_action_distribution "),a("Badge",{attrs:{text:"UncertainPolicies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_next_action_distribution",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"Distribution[D.T_agent[D.T_concurrency[D.T_event]]]"}}}),t._v(" "),a("p",[t._v("Get the probabilistic distribution of next action for the given observation (from the solver's current\npolicy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-16"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-16"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-16"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-16"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The probabilistic distribution of next action.")]),t._v(" "),a("h3",{attrs:{id:"is-policy-defined-for-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#is-policy-defined-for-6"}},[t._v("#")]),t._v(" _is_policy_defined_for "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_is_policy_defined_for",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"bool"}}}),t._v(" "),a("p",[t._v("Check whether the solver's current policy is defined for the given observation.")]),t._v(" "),a("h4",{attrs:{id:"parameters-17"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-17"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation to consider.")])]),t._v(" "),a("h4",{attrs:{id:"returns-17"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-17"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("True if the policy is defined for the given observation memory (False otherwise).")]),t._v(" "),a("h3",{attrs:{id:"sample-action-6"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#sample-action-6"}},[t._v("#")]),t._v(" _sample_action "),a("Badge",{attrs:{text:"Policies",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_sample_action",sig:{params:[{name:"self"},{name:"observation",annotation:"D.T_agent[D.T_observation]"}],return:"D.T_agent[D.T_concurrency[D.T_event]]"}}}),t._v(" "),a("p",[t._v("Sample an action for the given observation (from the solver's current policy).")]),t._v(" "),a("h4",{attrs:{id:"parameters-18"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#parameters-18"}},[t._v("#")]),t._v(" Parameters")]),t._v(" "),a("ul",[a("li",[a("strong",[t._v("observation")]),t._v(": The observation for which an action must be sampled.")])]),t._v(" "),a("h4",{attrs:{id:"returns-18"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-18"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("The sampled action.")])],1)}),[],!1,null,null,null);e.default=s.exports}}]);