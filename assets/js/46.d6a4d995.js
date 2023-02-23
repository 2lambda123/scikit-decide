(window.webpackJsonp=window.webpackJsonp||[]).push([[46],{559:function(e,s,t){"use strict";t.r(s);var a=t(38),i=Object(a.a)({},(function(){var e=this,s=e.$createElement,t=e._self._c||s;return t("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[t("h1",{attrs:{id:"builders-domain-scheduling-skills"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-scheduling-skills"}},[e._v("#")]),e._v(" builders.domain.scheduling.skills")]),e._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"custom-block-title"},[e._v("Domain specification")]),e._v(" "),t("skdecide-summary")],1),e._v(" "),t("h2",{attrs:{id:"withresourceskills"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#withresourceskills"}},[e._v("#")]),e._v(" WithResourceSkills")]),e._v(" "),t("p",[e._v("A domain must inherit this class if its resources (either resource types or resource units)\nhave different set of skills.")]),e._v(" "),t("h3",{attrs:{id:"find-one-ressource-to-do-one-task"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#find-one-ressource-to-do-one-task"}},[e._v("#")]),e._v(" find_one_ressource_to_do_one_task "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"find_one_ressource_to_do_one_task",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"mode",annotation:"int"}],return:"List[str]"}}}),e._v(" "),t("p",[e._v("For the common case when it is possible to do the task by one resource unit.\nFor general case, it might just return no possible ressource unit.")]),e._v(" "),t("h3",{attrs:{id:"get-all-resources-skills"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-resources-skills"}},[e._v("#")]),e._v(" get_all_resources_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_all_resources_skills",sig:{params:[{name:"self"}],return:"Dict[str, Dict[str, Any]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a resource type or resource unit\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {unit: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-all-tasks-skills"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-tasks-skills"}},[e._v("#")]),e._v(" get_all_tasks_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_all_tasks_skills",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, Dict[str, Any]]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a task\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {task: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-skills-names"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-names"}},[e._v("#")]),e._v(" get_skills_names "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_names",sig:{params:[{name:"self"}],return:"Set[str]"}}}),e._v(" "),t("p",[e._v("Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned\nby the get_all_resources_skills and get_all_tasks_skills functions.")]),e._v(" "),t("h3",{attrs:{id:"get-skills-of-resource"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-of-resource"}},[e._v("#")]),e._v(" get_skills_of_resource "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_of_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"}],return:"Dict[str, Any]"}}}),e._v(" "),t("p",[e._v("Return the skills of a given resource")]),e._v(" "),t("h3",{attrs:{id:"get-skills-of-task"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-of-task"}},[e._v("#")]),e._v(" get_skills_of_task "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_of_task",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"mode",annotation:"int"}],return:"Dict[str, Any]"}}}),e._v(" "),t("p",[e._v("Return the skill requirements for a given task")]),e._v(" "),t("h3",{attrs:{id:"get-all-resources-skills-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-resources-skills-2"}},[e._v("#")]),e._v(" _get_all_resources_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"_get_all_resources_skills",sig:{params:[{name:"self"}],return:"Dict[str, Dict[str, Any]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a resource type or resource unit\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {unit: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-all-tasks-skills-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-tasks-skills-2"}},[e._v("#")]),e._v(" _get_all_tasks_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"tip"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"_get_all_tasks_skills",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, Dict[str, Any]]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a task\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {task: {skill: (detail of skill)}}")]),e._v(" "),t("h2",{attrs:{id:"withoutresourceskills"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#withoutresourceskills"}},[e._v("#")]),e._v(" WithoutResourceSkills")]),e._v(" "),t("p",[e._v("A domain must inherit this class if no resources skills have to be considered.")]),e._v(" "),t("h3",{attrs:{id:"find-one-ressource-to-do-one-task-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#find-one-ressource-to-do-one-task-2"}},[e._v("#")]),e._v(" find_one_ressource_to_do_one_task "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"find_one_ressource_to_do_one_task",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"mode",annotation:"int"}],return:"List[str]"}}}),e._v(" "),t("p",[e._v("For the common case when it is possible to do the task by one resource unit.\nFor general case, it might just return no possible ressource unit.")]),e._v(" "),t("h3",{attrs:{id:"get-all-resources-skills-3"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-resources-skills-3"}},[e._v("#")]),e._v(" get_all_resources_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_all_resources_skills",sig:{params:[{name:"self"}],return:"Dict[str, Dict[str, Any]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a resource type or resource unit\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {unit: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-all-tasks-skills-3"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-tasks-skills-3"}},[e._v("#")]),e._v(" get_all_tasks_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_all_tasks_skills",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, Dict[str, Any]]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a task\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {task: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-skills-names-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-names-2"}},[e._v("#")]),e._v(" get_skills_names "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_names",sig:{params:[{name:"self"}],return:"Set[str]"}}}),e._v(" "),t("p",[e._v("Return a list of all skill names as a list of str. Skill names are defined in the 2 dictionaries returned\nby the get_all_resources_skills and get_all_tasks_skills functions.")]),e._v(" "),t("h3",{attrs:{id:"get-skills-of-resource-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-of-resource-2"}},[e._v("#")]),e._v(" get_skills_of_resource "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_of_resource",sig:{params:[{name:"self"},{name:"resource",annotation:"str"}],return:"Dict[str, Any]"}}}),e._v(" "),t("p",[e._v("Return the skills of a given resource")]),e._v(" "),t("h3",{attrs:{id:"get-skills-of-task-2"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-skills-of-task-2"}},[e._v("#")]),e._v(" get_skills_of_task "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"get_skills_of_task",sig:{params:[{name:"self"},{name:"task",annotation:"int"},{name:"mode",annotation:"int"}],return:"Dict[str, Any]"}}}),e._v(" "),t("p",[e._v("Return the skill requirements for a given task")]),e._v(" "),t("h3",{attrs:{id:"get-all-resources-skills-4"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-resources-skills-4"}},[e._v("#")]),e._v(" _get_all_resources_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"_get_all_resources_skills",sig:{params:[{name:"self"}],return:"Dict[str, Dict[str, Any]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a resource type or resource unit\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {unit: {skill: (detail of skill)}}")]),e._v(" "),t("h3",{attrs:{id:"get-all-tasks-skills-4"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#get-all-tasks-skills-4"}},[e._v("#")]),e._v(" _get_all_tasks_skills "),t("Badge",{attrs:{text:"WithResourceSkills",type:"warn"}})],1),e._v(" "),t("skdecide-signature",{attrs:{name:"_get_all_tasks_skills",sig:{params:[{name:"self"}],return:"Dict[int, Dict[str, Any]]"}}}),e._v(" "),t("p",[e._v("Return a nested dictionary where the first key is the name of a task\nand the second key is the name of a skill. The value defines the details of the skill.\nE.g. {task: {skill: (detail of skill)}}")])],1)}),[],!1,null,null,null);s.default=i.exports}}]);