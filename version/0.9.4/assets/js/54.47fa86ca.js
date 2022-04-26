(window.webpackJsonp=window.webpackJsonp||[]).push([[54],{568:function(e,a,t){"use strict";t.r(a);var r=t(38),s=Object(r.a)({},(function(){var e=this,a=e.$createElement,t=e._self._c||a;return t("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[t("h1",{attrs:{id:"builders-solver-parallelability"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#builders-solver-parallelability"}},[e._v("#")]),e._v(" builders.solver.parallelability")]),e._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),t("skdecide-summary")],1),e._v(" "),t("h2",{attrs:{id:"parallelsolver"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parallelsolver"}},[e._v("#")]),e._v(" ParallelSolver")]),e._v(" "),t("p",[e._v("A solver must inherit this class if it wants to call several cloned parallel domains in separate concurrent processes.\nThe solver is meant to be called either within a 'with' context statement, or to be cleaned up using the close() method.")]),e._v(" "),t("h3",{attrs:{id:"constructor"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[e._v("#")]),e._v(" Constructor "),t("Badge",{attrs:{text:"ParallelSolver",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"ParallelSolver",sig:{params:[{name:"domain_factory",annotation:"Callable[[], Domain]"},{name:"parallel",default:"False",annotation:"bool"},{name:"shared_memory_proxy",default:"None"}]}}}),e._v(" "),t("p",[e._v("Creates a parallelizable solver")]),e._v(" "),t("h4",{attrs:{id:"parameters"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#parameters"}},[e._v("#")]),e._v(" Parameters")]),e._v(" "),t("ul",[t("li",[t("strong",[e._v("domain_factory")]),e._v(": A callable with no argument returning the domain to solve (factory is the domain class if None).")]),e._v(" "),t("li",[t("strong",[e._v("parallel")]),e._v(": True if the solver is run in parallel mode.")]),e._v(" "),t("li",[t("strong",[e._v("shared_memory_proxy")]),e._v(": Shared memory proxy to use if not None, otherwise run piped parallel domains.")])]),e._v(" "),t("h3",{attrs:{id:"call-domain-method"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#call-domain-method"}},[e._v("#")]),e._v(" call_domain_method "),t("Badge",{attrs:{text:"ParallelSolver",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"call_domain_method",sig:{params:[{name:"self"},{name:"name"},{name:"*args"}]}}}),e._v(" "),t("p",[e._v("Calls a parallel domain's method.\nThis is the only way to get a domain method for a parallel domain.")]),e._v(" "),t("h3",{attrs:{id:"close"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#close"}},[e._v("#")]),e._v(" close "),t("Badge",{attrs:{text:"ParallelSolver",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"close",sig:{params:[{name:"self"}]}}}),e._v(" "),t("p",[e._v("Joins the parallel domains' processes.\nNot calling this method (or not using the 'with' context statement)\nresults in the solver forever waiting for the domain processes to exit.")]),e._v(" "),t("h3",{attrs:{id:"get-domain"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-domain"}},[e._v("#")]),e._v(" get_domain "),t("Badge",{attrs:{text:"ParallelSolver",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_domain",sig:{params:[{name:"self"}]}}}),e._v(" "),t("p",[e._v("Returns the domain, optionally creating a parallel domain if not already created.")]),e._v(" "),t("h3",{attrs:{id:"initialize"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#initialize"}},[e._v("#")]),e._v(" _initialize "),t("Badge",{attrs:{text:"ParallelSolver",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"_initialize",sig:{params:[{name:"self"}]}}}),e._v(" "),t("p",[e._v("Launches the parallel domains.\nThis method requires to have previously recorded the self._domain_factory (e.g. after calling _init_solve),\nthe set of lambda functions passed to the solver's constructor (e.g. heuristic lambda for heuristic-based solvers),\nand whether the parallel domain jobs should notify their status via the IPC protocol (required when interacting with\nother programming languages like C++)")])],1)}),[],!1,null,null,null);a.default=s.exports}}]);