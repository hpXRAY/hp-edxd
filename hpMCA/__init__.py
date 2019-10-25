# -*- coding: utf8 -*-

# DISCLAIMER
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Principal author: R. Hrubiak (hrubiak@anl.gov)
# Copyright (C) 2018-2019 ANL, Lemont, USA

# Based on code from Dioptas - GUI program for fast processing of 2D X-ray diffraction data

from __future__ import absolute_import

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

if __version__ == "0+unknown":
    __version__ = "0.2.0"

import sys
import os
import time

from qtpy import QtWidgets

resources_path = os.path.join(os.path.dirname(__file__), 'resources')
#calibrants_path = os.path.join(resources_path, 'calibrants')
icons_path = os.path.join(resources_path, 'icons')
#data_path = os.path.join(resources_path, 'data')
#style_path = os.path.join(resources_path, 'style')

from ._desktop_shortcuts import make_shortcut

from .widgets.UtilityWidgets import ErrorMessageBox





def excepthook(exc_type, exc_value, traceback_obj):
    """
    Global function to catch unhandled exceptions. This function will result in an error dialog which displays the
    error information.

    :param exc_type: exception type
    :param exc_value: exception value
    :param traceback_obj: traceback object
    :return:
    """
    separator = '-' * 80
    log_file = "error.log"
    notice = \
        """An unhandled exception occurred. Please report the bug under:\n """ \
        """\t%s\n""" \
        """or via email to:\n\t <%s>.\n\n""" \
        """A log has been written to "%s".\n\nError information:\n""" % \
        (" ",
         "hrubiak@anl.gov",
         os.path.join(os.path.dirname(__file__), log_file))
    version_info = '\n'.join((separator, "hpMCA Version: %s" % dioptas_version))
    time_string = time.strftime("%Y-%m-%d, %H:%M:%S")
    tb_info_file = StringIO()
    traceback.print_tb(traceback_obj, None, tb_info_file)
    tb_info_file.seek(0)
    tb_info = tb_info_file.read()
    errmsg = '%s: \n%s' % (str(exc_type), str(exc_value))
    sections = [separator, time_string, separator, errmsg, separator, tb_info]
    msg = '\n'.join(sections)
    try:
        f = open(log_file, "w")
        f.write(msg)
        f.write(version_info)
        f.close()
    except IOError:
        pass
    errorbox = ErrorMessageBox()
    errorbox.setText(str(notice) + str(msg) + str(version_info))
    errorbox.exec_()


def main():
    app = QtWidgets.QApplication([])

    from hpMCA.controllers.hpmca_controller import hpMCA
    app.aboutToQuit.connect(app.deleteLater)

    controller = hpMCA(app)
    controller.widget.show()

    # autoload a file, using for debugging
    #controller.openFile(filename='resources/20181010-Au-wire-50um-15deg.hpmca')
    #controller.phase_controller.add_btn_click_callback(filenames=['JCPDS/Metals/au.jcpds'])
    #controller.phase_controller.add_btn_click_callback(filenames=['JCPDS/Oxides/mgo.jcpds'])

    return app.exec_()