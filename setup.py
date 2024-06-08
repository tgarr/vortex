#!/usr/bin/env python3

import os
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI


OBJECT_POOLS_LIST = "setup/object_pools.list"

SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }


def main(argv):

    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    #basepath = os.path.dirname(argv[0])
    basepath = "."

    # create object pools
    print("Creating object pools ...")
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    with open(fpath,"r") as list_file:
        for line in list_file:
            fields = line.strip().split(" ")
            pool_path = fields[0]
            subgroup_type = fields[1]
            subgroup_index = int(fields[2])
            if len(fields) >= 4:
                affinity_set_regex = fields[3].strip()
                print(f"AFFINITY: {pool_path} {affinity_set_regex}")
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],subgroup_index,affinity_set_regex=affinity_set_regex)
                if res:
                    res.get_result()
            else:
                print(f"  {pool_path} {subgroup_type} {subgroup_index}")
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],subgroup_index)
                if res:
                    res.get_result()
            # time.sleep(0.5)


    print("Done!")

if __name__ == "__main__":
    main(sys.argv)

