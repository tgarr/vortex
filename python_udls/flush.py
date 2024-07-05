#!/usr/bin/env python3
import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic

class FlushUDL(UserDefinedLogic):
     def __init__(self,conf_str):
          self.capi = ServiceClientAPI()
          self.tl = TimestampLogger()
          self.my_id = self.capi.get_my_id()

     def ocdpo_handler(self,**kwargs):
          key = kwargs["key"]
          print("FlushUDL: Received ocdpo_handler call")
          self.tl.flush(f"node{self.my_id}_udls_timestamp.dat")


     def __del__(self):
          pass