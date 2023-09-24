(window.webpackJsonp=window.webpackJsonp||[]).push([[107],{620:function(e,t,o){"use strict";o.r(t);var n=o(38),a=Object(n.a)({},(function(){var e=this,t=e.$createElement,o=e._self._c||t;return o("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[o("h1",{attrs:{id:"utils"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#utils"}},[e._v("#")]),e._v(" utils")]),e._v(" "),o("p",[e._v("This module contains utility functions.")]),e._v(" "),o("div",{staticClass:"custom-block tip"},[o("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),o("skdecide-summary")],1),e._v(" "),o("h2",{attrs:{id:"rollout"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#rollout"}},[e._v("#")]),e._v(" rollout")]),e._v(" "),o("skdecide-signature",{attrs:{name:"rollout",sig:{params:[{name:"domain",annotation:"Domain"},{name:"solver",default:"None",annotation:"Optional[Solver]"},{name:"from_memory",default:"None",annotation:"Optional[D.T_memory[D.T_state]]"},{name:"from_action",default:"None",annotation:"Optional[D.T_agent[D.T_concurrency[D.T_event]]]"},{name:"num_episodes",default:"1",annotation:"int"},{name:"max_steps",default:"None",annotation:"Optional[int]"},{name:"render",default:"True",annotation:"bool"},{name:"max_framerate",default:"None",annotation:"Optional[float]"},{name:"verbose",default:"True",annotation:"bool"},{name:"action_formatter",default:"<lambda function>",annotation:"Optional[Callable[[D.T_event], str]]"},{name:"outcome_formatter",default:"<lambda function>",annotation:"Optional[Callable[[EnvironmentOutcome], str]]"},{name:"save_result_directory",default:"None",annotation:"str"}],return:"str"}}}),e._v(" "),o("p",[e._v("This method will run one or more episodes in a domain according to the policy of a solver.")]),e._v(" "),o("h4",{attrs:{id:"parameters"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),o("ul",[o("li",[o("strong",[e._v("domain")]),e._v(": The domain in which the episode(s) will be run.")]),e._v(" "),o("li",[o("strong",[e._v("solver")]),e._v(": The solver whose policy will select actions to take (if None, a random policy is used).")]),e._v(" "),o("li",[o("strong",[e._v("from_memory")]),e._v(": The memory or state to consider as rollout starting point (if None, the domain is reset first).")]),e._v(" "),o("li",[o("strong",[e._v("from_action")]),e._v(": The last applied action when from_memory is used (if necessary for initial observation computation).")]),e._v(" "),o("li",[o("strong",[e._v("num_episodes")]),e._v(": The number of episodes to run.")]),e._v(" "),o("li",[o("strong",[e._v("max_steps")]),e._v(": The maximum number of steps for each episode (if None, no limit is set).")]),e._v(" "),o("li",[o("strong",[e._v("render")]),e._v(": Whether to render the episode(s) during rollout if the domain is renderable.")]),e._v(" "),o("li",[o("strong",[e._v("max_framerate")]),e._v(": The maximum number of steps/renders per second (if None, steps/renders are never slowed down).")]),e._v(" "),o("li",[o("strong",[e._v("verbose")]),e._v(": Whether to print information to the console during rollout.")]),e._v(" "),o("li",[o("strong",[e._v("action_formatter")]),e._v(": The function transforming actions in the string to print (if None, no print).")]),e._v(" "),o("li",[o("strong",[e._v("outcome_formatter")]),e._v(": The function transforming EnvironmentOutcome objects in the string to print (if None, no print).")]),e._v(" "),o("li",[o("strong",[e._v("save_result")]),e._v(": Directory in which state visited, actions applied and Transition Value are saved to json.")])]),e._v(" "),o("h2",{attrs:{id:"rollout-episode"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#rollout-episode"}},[e._v("#")]),e._v(" rollout_episode")]),e._v(" "),o("skdecide-signature",{attrs:{name:"rollout_episode",sig:{params:[{name:"domain",annotation:"Domain"},{name:"solver",default:"None",annotation:"Optional[Union[Solver, Policies]]"},{name:"from_memory",default:"None",annotation:"Optional[D.T_memory[D.T_state]]"},{name:"from_action",default:"None",annotation:"Optional[D.T_agent[D.T_concurrency[D.T_event]]]"},{name:"num_episodes",default:"1",annotation:"int"},{name:"max_steps",default:"None",annotation:"Optional[int]"},{name:"render",default:"True",annotation:"bool"},{name:"max_framerate",default:"None",annotation:"Optional[float]"},{name:"verbose",default:"True",annotation:"bool"},{name:"action_formatter",default:"None",annotation:"Optional[Callable[[D.T_event], str]]"},{name:"outcome_formatter",default:"None",annotation:"Optional[Callable[[EnvironmentOutcome], str]]"},{name:"save_result_directory",default:"None",annotation:"str"}],return:"Tuple[List[D.T_observation], List[D.T_event], List[D.T_value]]"}}}),e._v(" "),o("p",[e._v("This method will run one or more episodes in a domain according to the policy of a solver.")]),e._v(" "),o("h4",{attrs:{id:"parameters-2"}},[o("a",{staticClass:"header-anchor",attrs:{href:"#parameters-2"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),o("ul",[o("li",[o("strong",[e._v("domain")]),e._v(": The domain in which the episode(s) will be run.")]),e._v(" "),o("li",[o("strong",[e._v("solver")]),e._v(": The solver whose policy will select actions to take (if None, a random policy is used).")]),e._v(" "),o("li",[o("strong",[e._v("from_memory")]),e._v(": The memory or state to consider as rollout starting point (if None, the domain is reset first).")]),e._v(" "),o("li",[o("strong",[e._v("from_action")]),e._v(": The last applied action when from_memory is used (if necessary for initial observation computation).")]),e._v(" "),o("li",[o("strong",[e._v("num_episodes")]),e._v(": The number of episodes to run.")]),e._v(" "),o("li",[o("strong",[e._v("max_steps")]),e._v(": The maximum number of steps for each episode (if None, no limit is set).")]),e._v(" "),o("li",[o("strong",[e._v("render")]),e._v(": Whether to render the episode(s) during rollout if the domain is renderable.")]),e._v(" "),o("li",[o("strong",[e._v("max_framerate")]),e._v(": The maximum number of steps/renders per second (if None, steps/renders are never slowed down).")]),e._v(" "),o("li",[o("strong",[e._v("verbose")]),e._v(": Whether to print information to the console during rollout.")]),e._v(" "),o("li",[o("strong",[e._v("action_formatter")]),e._v(": The function transforming actions in the string to print (if None, no print).")]),e._v(" "),o("li",[o("strong",[e._v("outcome_formatter")]),e._v(": The function transforming EnvironmentOutcome objects in the string to print (if None, no print).")]),e._v(" "),o("li",[o("strong",[e._v("save_result")]),e._v(": Directory in which state visited, actions applied and Transition Value are saved to json.")])])],1)}),[],!1,null,null,null);t.default=a.exports}}]);