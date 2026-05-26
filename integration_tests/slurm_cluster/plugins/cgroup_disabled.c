#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifndef SLURM_PLUGIN_VERSION
#define SLURM_PLUGIN_VERSION 0
#endif

const char plugin_name[] = "Cgroup disabled no-op plugin";
const char plugin_type[] = "disabled";
const uint32_t plugin_version = SLURM_PLUGIN_VERSION;

int cgroup_p_initialize(int sub) { return 0; }
int cgroup_p_system_create(int sub) { return 0; }
int cgroup_p_system_addto(int sub, pid_t *pids, int npids) { return 0; }
int cgroup_p_system_destroy(int sub) { return 0; }
int cgroup_p_step_create(int sub, void *step) { return 0; }
int cgroup_p_step_addto(int sub, pid_t *pids, int npids) { return 0; }
int cgroup_p_step_get_pids(pid_t **pids, int *npids) { return 0; }
int cgroup_p_step_suspend(void) { return 0; }
int cgroup_p_step_resume(void) { return 0; }
int cgroup_p_step_destroy(int sub) { return 0; }
bool cgroup_p_has_pid(pid_t pid) { return false; }
void *cgroup_p_constrain_get(int sub, int level) { return NULL; }
int cgroup_p_constrain_set(int sub, int level, void *limits) { return 0; }
int cgroup_p_constrain_apply(int sub, int level, uint32_t task_id) { return 0; }
int cgroup_p_step_start_oom_mgr(void *step) { return 0; }
void *cgroup_p_step_stop_oom_mgr(void *step) { return NULL; }
int cgroup_p_task_addto(int sub, void *step, pid_t pid, uint32_t task_id) { return 0; }
void *cgroup_p_task_get_acct_data(uint32_t task_id) { return NULL; }
void *cgroup_p_job_get_acct_data(void) { return NULL; }
long int cgroup_p_get_acct_units(void) { return 1; }
bool cgroup_p_has_feature(int feature) { return false; }
char *cgroup_p_get_scope_path(void) { return NULL; }
int cgroup_p_bpf_fsopen(void) { return -1; }
int cgroup_p_bpf_fsconfig(int fd) { return -1; }
int cgroup_p_bpf_create_token(int fd) { return -1; }
void cgroup_p_bpf_set_token(int fd) {}
int cgroup_p_bpf_get_token(void) { return -1; }
int cgroup_p_setup_scope(char *scope_path) { return 0; }
int cgroup_p_signal(int signal) { return 0; }
char *cgroup_p_get_task_empty_event_path(uint32_t taskid, bool *on_modify) { return NULL; }
int cgroup_p_is_task_empty(uint32_t taskid) { return 1; }
