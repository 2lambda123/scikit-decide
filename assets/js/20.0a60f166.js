(window.webpackJsonp=window.webpackJsonp||[]).push([[20],{619:function(e,t,n){"use strict";n.r(t);var a=n(38),s=Object(a.a)({},(function(){var e=this,t=e.$createElement,n=e._self._c||t;return n("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[n("h1",{attrs:{id:"installation"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#installation"}},[e._v("#")]),e._v(" Installation")]),e._v(" "),n("p",[e._v("## Prerequisites")]),e._v(" "),n("p",[e._v("### Minizinc 2.6+")]),e._v(" "),n("p",[e._v("You need to install "),n("a",{attrs:{href:"https://www.minizinc.org/",target:"_blank",rel:"noopener noreferrer"}},[e._v("minizinc"),n("OutboundLink")],1),e._v(" (version greater than 2.6) and update the "),n("code",[e._v("PATH")]),e._v(" environment variable\nso that it can be found by Python. See "),n("a",{attrs:{href:"https://www.minizinc.org/doc-latest/en/installation.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("minizinc documentation"),n("OutboundLink")],1),e._v(" for more details.")]),e._v(" "),n("h3",{attrs:{id:"python-3-7-environment"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#python-3-7-environment"}},[e._v("#")]),e._v(" Python 3.7+ environment")]),e._v(" "),n("p",[e._v("The use of a virtual environment for scikit-decide is recommended, and you will need to ensure that the environment use a Python version greater than 3.7.\nThis can be achieved either by using "),n("a",{attrs:{href:"https://docs.conda.io/en/latest/",target:"_blank",rel:"noopener noreferrer"}},[e._v("conda"),n("OutboundLink")],1),e._v(" or by using "),n("a",{attrs:{href:"https://github.com/pyenv/pyenv",target:"_blank",rel:"noopener noreferrer"}},[e._v("pyenv"),n("OutboundLink")],1),e._v(" (or "),n("a",{attrs:{href:"https://github.com/pyenv-win/pyenv-win",target:"_blank",rel:"noopener noreferrer"}},[e._v("pyenv-win"),n("OutboundLink")],1),e._v(" on windows)\nand "),n("a",{attrs:{href:"https://docs.python.org/fr/3/library/venv.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("venv"),n("OutboundLink")],1),e._v(" module.")]),e._v(" "),n("p",[e._v("The following examples show how to create a virtual environment with Python version 3.8.11 with the mentioned methods.")]),e._v(" "),n("h4",{attrs:{id:"with-conda-all-platforms"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#with-conda-all-platforms"}},[e._v("#")]),e._v(" With conda (all platforms)")]),e._v(" "),n("div",{staticClass:"language-shell extra-class"},[n("pre",{pre:!0,attrs:{class:"language-shell"}},[n("code",[e._v("conda create -n skdecide "),n("span",{pre:!0,attrs:{class:"token assign-left variable"}},[e._v("python")]),n("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),n("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.8")]),e._v(".11\nconda activate skdecide\n")])])]),n("h4",{attrs:{id:"with-pyenv-venv-linux-macos"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#with-pyenv-venv-linux-macos"}},[e._v("#")]),e._v(" With pyenv + venv (Linux/MacOS)")]),e._v(" "),n("div",{staticClass:"language-shell extra-class"},[n("pre",{pre:!0,attrs:{class:"language-shell"}},[n("code",[e._v("pyenv "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" "),n("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.8")]),e._v(".11\npyenv shell "),n("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.8")]),e._v(".11\npython -m venv skdecide-venv\n"),n("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("source")]),e._v(" skdecide-venv\n")])])]),n("h4",{attrs:{id:"with-pyenv-win-venv-windows"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#with-pyenv-win-venv-windows"}},[e._v("#")]),e._v(" With pyenv-win + venv (Windows)")]),e._v(" "),n("div",{staticClass:"language-shell extra-class"},[n("pre",{pre:!0,attrs:{class:"language-shell"}},[n("code",[e._v("pyenv "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" "),n("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.8")]),e._v(".11\npyenv shell "),n("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.8")]),e._v(".11\npython -m venv skdecide-venv\nskdecide-venv"),n("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("\\")]),e._v("Scripts"),n("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("\\")]),e._v("activate\n")])])]),n("h2",{attrs:{id:"install-scikit-decide-library"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#install-scikit-decide-library"}},[e._v("#")]),e._v(" Install scikit-decide library")]),e._v(" "),n("h3",{attrs:{id:"full-install-recommended"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#full-install-recommended"}},[e._v("#")]),e._v(" Full install [Recommended]")]),e._v(" "),n("p",[e._v("Install scikit-decide library from PyPI with all dependencies required by domains/solvers in the hub (scikit-decide catalog).")]),e._v(" "),n("div",{staticClass:"language-shell extra-class"},[n("pre",{pre:!0,attrs:{class:"language-shell"}},[n("code",[e._v("pip "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U pip\npip "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U scikit-decide"),n("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("[")]),e._v("all"),n("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("]")]),e._v("\n")])])]),n("h3",{attrs:{id:"minimal-install"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#minimal-install"}},[e._v("#")]),e._v(" Minimal install")]),e._v(" "),n("p",[e._v("Alternatively you can choose to only install the core library, which is enough if you intend to create your own domain and solver.")]),e._v(" "),n("div",{staticClass:"language-shell extra-class"},[n("pre",{pre:!0,attrs:{class:"language-shell"}},[n("code",[e._v("pip "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U pip\npip "),n("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U scikit-decide\n")])])])])}),[],!1,null,null,null);t.default=s.exports}}]);