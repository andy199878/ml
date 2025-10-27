#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TASK_READY,
    TASK_RUNNING,
    TASK_BLOCKED,
    TASK_FINISHED
} TaskState;

typedef struct {
    jmp_buf context;
    int priority;
    TaskState state;
    const char *name;
    int work_count;
    int max_work;
    int initialized;
} Task;

#define MAX_TASKS 3
Task tasks[MAX_TASKS];
Task *current_task = NULL;
int task_count = 0;
jmp_buf scheduler_context;

// 找到最高優先權 READY 任務
Task* find_highest_priority_task() {
    Task *highest = NULL;
    for (int i = 0; i < task_count; i++) {
        if (tasks[i].state == TASK_READY) {
            if (!highest || tasks[i].priority > highest->priority) {
                highest = &tasks[i];
            }
        }
    }
    return highest;
}

// 任務讓出 CPU
void task_yield() {
    if (!current_task) return;

    Task *prev = current_task;
    Task *next = find_highest_priority_task();

    if (!next || next == prev) return;

    int val = setjmp(prev->context);
    if (val == 0) {
        printf("[SCHEDULER] Context switch: %s (P%d) -> %s (P%d)\n",
               prev->name, prev->priority, next->name, next->priority);
        prev->state = TASK_READY;
        current_task = next;
        next->state = TASK_RUNNING;
        longjmp(next->context, 1);
    }
}

// ISR 只喚醒阻塞任務，不切換
void simulate_interrupt(int id) {
    printf("\n[ISR] Interrupt %d occurred!\n", id);
    for (int i = 0; i < task_count; i++) {
        if (tasks[i].state == TASK_BLOCKED) {
            tasks[i].state = TASK_READY;
            printf("[ISR] Task %s unblocked\n", tasks[i].name);
        }
    }
}

// 任務完成
void task_finish(Task *t) {
    t->state = TASK_FINISHED;
    printf("[TASK] %s finished\n", t->name);
    current_task = NULL;
    longjmp(scheduler_context, 1);
}

// 任務函式
void task1_func() {
    if (setjmp(tasks[0].context) == 0) { tasks[0].initialized = 1; return; }

    printf("[TASK1] Started (P%d)\n", tasks[0].priority);
    while (tasks[0].work_count < tasks[0].max_work) {
        printf("[TASK1] Working... (%d/%d)\n", tasks[0].work_count+1, tasks[0].max_work);
        for (volatile int i=0; i<50000000; i++);
        tasks[0].work_count++;
        task_yield();
    }
    task_finish(&tasks[0]);
}

void task2_func() {
    if (setjmp(tasks[1].context) == 0) { tasks[1].initialized = 1; return; }

    printf("[TASK2] Started (P%d)\n", tasks[1].priority);
    while (tasks[1].work_count < tasks[1].max_work) {
        printf("[TASK2] Working... (%d/%d)\n", tasks[1].work_count+1, tasks[1].max_work);
        for (volatile int i=0; i<50000000; i++);
        tasks[1].work_count++;

        if (tasks[1].work_count == 1) {
            printf("[TASK2] Blocking itself...\n");
            tasks[1].state = TASK_BLOCKED;
            task_yield();
            printf("[TASK2] Resumed after unblock\n");
        } else {
            task_yield();
        }
    }
    task_finish(&tasks[1]);
}

void task3_func() {
    if (setjmp(tasks[2].context) == 0) { tasks[2].initialized = 1; return; }

    printf("[TASK3] Started (P%d)\n", tasks[2].priority);
    while (tasks[2].work_count < tasks[2].max_work) {
        printf("[TASK3] Working... (%d/%d)\n", tasks[2].work_count+1, tasks[2].max_work);
        for (volatile int i=0; i<50000000; i++);
        tasks[2].work_count++;

        if (tasks[2].work_count == 2)
            simulate_interrupt(1);

        task_yield();
    }
    task_finish(&tasks[2]);
}

// 初始化任務
void init_task(int id, const char *name, int prio, int max_work, void (*func)()) {
    memset(&tasks[id],0,sizeof(Task));
    tasks[id].priority = prio;
    tasks[id].state = TASK_READY;
    tasks[id].name = name;
    tasks[id].max_work = max_work;
    func(); // 初始化上下文
    task_count++;
}

// 啟動 scheduler
void start_scheduler() {
    printf("\n=== Scheduler Started ===\n\n");
    while (1) {
        if (setjmp(scheduler_context) != 0)
            printf("[SCHEDULER] Task completed, checking next...\n");

        current_task = find_highest_priority_task();
        if (!current_task) {
            printf("[SCHEDULER] No more tasks to run!\n");
            break;
        }

        current_task->state = TASK_RUNNING;
        longjmp(current_task->context,1);
    }
}

int main() {
    printf("=== Cooperative Task Scheduler Demo ===\nPriority: Higher number = Higher priority\n\n");

    init_task(0,"Task1",1,3,task1_func);
    init_task(1,"Task2",3,3,task2_func);
    init_task(2,"Task3",2,3,task3_func);

    start_scheduler();
    printf("\n=== All Tasks Completed ===\n");
    return 0;
}
