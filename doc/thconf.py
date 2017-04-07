from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList


class THConfDirective(Directive):
    def format_conf(self, cv, vl):
        env = self.state.document.settings.env
        vl.append(".. attribute:: %s" % (cv.fullname,), self.lineno, 0)

    def run(self):
        import theano
        env = self.state.document.settings.env
        vl = ViewList()
        for cv in sorted(theano.configparser._config_var_list,
                         key=lambda cv: cv.fullname):
            self.format_conf(cv, vl)
        return self.state.nested_parse(vl, 0, )


def setup(app):
    app.add_directive('thconf', THConfDirective)

    return {'version': '0.1'}

