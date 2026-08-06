"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2021-01-20 20:12:13
MODIFIED: 2021-02-03 14:04:45
"""
import acl
from .acllite_logger import log_info
from .acllite_resource_list import resource_list
from . import acllite_utils as utils


class AclLiteResource(object):
    """
    AclLiteResource
    """

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self):
        """
        init resource
        """
        log_info("init resource stage:")
        ret = acl.init()
        utils.check_ret("acl.init", ret)

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        log_info("Init resource success")

    def __del__(self):
        if getattr(self, "_skip_destroy", False):
            return
        log_info("acl resource release all resource")
        resource_list.destroy()
        if self.stream:
            log_info("acl resource release stream")
            acl.rt.destroy_stream(self.stream)

        if self.context:
            log_info("acl resource release context")
            acl.rt.destroy_context(self.context)

        log_info("Reset acl device ", self.device_id)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        log_info("Release acl resource success")
