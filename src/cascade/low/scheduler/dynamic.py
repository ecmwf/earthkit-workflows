"""
Simple Dynamic Scheduler
 - won't necessarily schedule everything -- only so that all workers are (reasonably) occupied
 - firstly, tries to allocate tasks that fit and won't cause any data transfer or OOM
 - secondly, finds the closest leaf and assigns it to # TODO
"""
