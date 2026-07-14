use super::BOX_FUTURE_THRESHOLD;
use crate::runtime::blocking::BlockingPool;
use crate::runtime::scheduler::CurrentThread;
use crate::runtime::{context, EnterGuard, Handle};
use crate::task::JoinHandle;
use crate::util::error::RUNTIME_SHUTTING_DOWN_ERROR;
use crate::util::trace::SpawnMeta;

use std::future::Future;
use std::io;
use std::mem;
use std::time::Duration;

cfg_rt_multi_thread! {
    use crate::runtime::Builder;
    use crate::runtime::scheduler::MultiThread;
}

/// The Tokio runtime.
///
/// The runtime provides an I/O driver, task scheduler, [timer], and
/// blocking pool, necessary for running asynchronous tasks.
///
/// Instances of `Runtime` can be created using [`new`], or [`Builder`].
/// However, most users will use the [`#[tokio::main]`][main] annotation on
/// their entry point instead.
///
/// See [module level][mod] documentation for more details.
///
/// # Shutdown
///
/// Shutting down the runtime is done by dropping the value, or calling
/// [`shutdown_background`] or [`shutdown_timeout`].
///
/// Tasks spawned through [`Runtime::spawn`] keep running until they yield.
/// Then they are dropped. They are not *guaranteed* to run to completion, but
/// *might* do so if they do not yield until completion.
///
/// Blocking functions spawned through [`Runtime::spawn_blocking`] keep running
/// until they return.
///
/// The thread initiating the shutdown blocks until all spawned work has been
/// stopped. This can take an indefinite amount of time. The `Drop`
/// implementation waits forever for this.
///
/// The [`shutdown_background`] and [`shutdown_timeout`] methods can be used if
/// waiting forever is undesired. When the timeout is reached, spawned work that
/// did not stop in time and threads running it are leaked. The work continues
/// to run until one of the stopping conditions is fulfilled, but the thread
/// initiating the shutdown is unblocked.
///
/// Once the runtime has been dropped, any outstanding I/O resources bound to
/// it will no longer function. Calling any method on them will result in an
/// error.
///
/// # Sharing
///
/// There are several ways to establish shared access to a Tokio runtime:
///
///  * Using an <code>[Arc]\<Runtime></code>.
///  * Using a [`Handle`].
///  * Entering the runtime context.
///
/// Using an <code>[Arc]\<Runtime></code> or [`Handle`] allows you to do various
/// things with the runtime such as spawning new tasks or entering the runtime
/// context. Both types can be cloned to create a new handle that allows access
/// to the same runtime. By passing clones into different tasks or threads, you
/// will be able to access the runtime from those tasks or threads.
///
/// The difference between <code>[Arc]\<Runtime></code> and [`Handle`] is that
/// an <code>[Arc]\<Runtime></code> will prevent the runtime from shutting down,
/// whereas a [`Handle`] does not prevent that. This is because shutdown of the
/// runtime happens when the destructor of the `Runtime` object runs.
///
/// Calls to [`shutdown_background`] and [`shutdown_timeout`] require exclusive
/// ownership of the `Runtime` type. When using an <code>[Arc]\<Runtime></code>,
/// this can be achieved via [`Arc::try_unwrap`] when only one strong count
/// reference is left over.
///
/// The runtime context is entered using the [`Runtime::enter`] or
/// [`Handle::enter`] methods, which use a thread-local variable to store the
/// current runtime. Whenever you are inside the runtime context, methods such
/// as [`tokio::spawn`] will use the runtime whose context you are inside.
///
/// [timer]: crate::time
/// [mod]: index.html
/// [`new`]: method@Self::new
/// [`Builder`]: struct@Builder
/// [`Handle`]: struct@Handle
/// [main]: macro@crate::main
/// [`tokio::spawn`]: crate::spawn
/// [`Arc::try_unwrap`]: std::sync::Arc::try_unwrap
/// [Arc]: std::sync::Arc
/// [`shutdown_background`]: method@Runtime::shutdown_background
/// [`shutdown_timeout`]: method@Runtime::shutdown_timeout
#[derive(Debug)]
pub struct Runtime {
    /// Task scheduler
    scheduler: Scheduler,

    /// Handle to runtime, also contains driver handles
    handle: Handle,

    /// Blocking pool handle, used to signal shutdown
    blocking_pool: BlockingPool,
}

/// The flavor of a `Runtime`.
///
/// This is the return type for [`Handle::runtime_flavor`](crate::runtime::Handle::runtime_flavor()).
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeFlavor {
    /// The flavor that executes all tasks on the current thread.
    CurrentThread,
    /// The flavor that executes tasks across multiple threads.
    MultiThread,
}

/// The runtime scheduler is either a multi-thread or a current-thread executor.
#[derive(Debug)]
pub(super) enum Scheduler {
    /// Execute all tasks on the current-thread.
    CurrentThread(CurrentThread),

    /// Execute tasks across multiple threads.
    #[cfg(feature = "rt-multi-thread")]
    MultiThread(MultiThread),
}

impl Runtime {
    pub(super) fn from_parts(
        scheduler: Scheduler,
        handle: Handle,
        blocking_pool: BlockingPool,
    ) -> Runtime {
        Runtime {
            scheduler,
            handle,
            blocking_pool,
        }
    }

    /// Creates a new runtime instance with default configuration values.
    ///
    /// This results in the multi threaded scheduler, I/O driver, and time driver being
    /// initialized.
    ///
    /// Most applications will not need to call this function directly. Instead,
    /// they will use the  [`#[tokio::main]` attribute][main]. When a more complex
    /// configuration is necessary, the [runtime builder] may be used.
    ///
    /// See [module level][mod] documentation for more details.
    ///
    /// # Examples
    ///
    /// Creating a new `Runtime` with default configuration values.
    ///
    /// ```
    /// use tokio::runtime::Runtime;
    ///
    /// let rt = Runtime::new()
    ///     .unwrap();
    ///
    /// // Use the runtime...
    /// ```
    ///
    /// [mod]: index.html
    /// [main]: ../attr.main.html
    /// [threaded scheduler]: index.html#threaded-scheduler
    /// [runtime builder]: crate::runtime::Builder
    #[cfg(feature = "rt-multi-thread")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rt-multi-thread")))]
    pub fn new() -> std::io::Result<Runtime> {
        Builder::new_multi_thread().enable_all().build()
    }

    /// Returns a handle to the runtime's spawner.
    ///
    /// The returned handle can be used to spawn tasks that run on this runtime, and can
    /// be cloned to allow moving the `Handle` to other threads.
    ///
    /// Calling [`Handle::block_on`] on a handle to a `current_thread` runtime is error-prone.
    /// Refer to the documentation of [`Handle::block_on`] for more.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    ///
    /// let rt = Runtime::new()
    ///     .unwrap();
    ///
    /// let handle = rt.handle();
    ///
    /// // Use the handle...
    /// # }
    /// ```
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// Spawns a future onto the Tokio runtime.
    ///
    /// This spawns the given future onto the runtime's executor, usually a
    /// thread pool. The thread pool is then responsible for polling the future
    /// until it completes.
    ///
    /// The provided future will start running in the background immediately
    /// when `spawn` is called, even if you don't await the returned
    /// `JoinHandle` (assuming that the runtime [is running][running-runtime]).
    ///
    /// See [module level][mod] documentation for more details.
    ///
    /// [mod]: index.html
    /// [running-runtime]: index.html#driving-the-runtime
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    ///
    /// # fn dox() {
    /// // Create the runtime
    /// let rt = Runtime::new().unwrap();
    ///
    /// // Spawn a future onto the runtime
    /// rt.spawn(async {
    ///     println!("now running on a worker thread");
    /// });
    /// # }
    /// # }
    /// ```
    #[track_caller]
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let fut_size = mem::size_of::<F>();
        if fut_size > BOX_FUTURE_THRESHOLD {
            self.handle
                .spawn_named(Box::pin(future), SpawnMeta::new_unnamed(fut_size))
        } else {
            self.handle
                .spawn_named(future, SpawnMeta::new_unnamed(fut_size))
        }
    }

    /// Runs the provided function on an executor dedicated to blocking operations.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    ///
    /// # fn dox() {
    /// // Create the runtime
    /// let rt = Runtime::new().unwrap();
    ///
    /// // Spawn a blocking function onto the runtime
    /// rt.spawn_blocking(|| {
    ///     println!("now running on a worker thread");
    /// });
    /// # }
    /// # }
    /// ```
    #[track_caller]
    pub fn spawn_blocking<F, R>(&self, func: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.handle.spawn_blocking(func)
    }

    /// Runs a future to completion on the Tokio runtime. This is the
    /// runtime's entry point.
    ///
    /// This runs the given future on the current thread, blocking until it is
    /// complete, and yielding its resolved result. Any tasks or timers
    /// which the future spawns internally will be executed on the runtime.
    ///
    /// # Non-worker future
    ///
    /// Note that the future required by this function does not run as a
    /// worker. The expectation is that other tasks are spawned by the future here.
    /// Awaiting on other futures from the future provided here will not
    /// perform as fast as those spawned as workers.
    ///
    /// # Multi thread scheduler
    ///
    /// When the multi thread scheduler is used this will allow futures
    /// to run within the io driver and timer context of the overall runtime.
    ///
    /// Any spawned tasks will continue running after `block_on` returns.
    ///
    /// # Current thread scheduler
    ///
    /// When the current thread scheduler is enabled `block_on`
    /// can be called concurrently from multiple threads. The first call
    /// will take ownership of the io and timer drivers. This means
    /// other threads which do not own the drivers will hook into that one.
    /// When the first `block_on` completes, other threads will be able to
    /// "steal" the driver to allow continued execution of their futures.
    ///
    /// Any spawned tasks will be suspended after `block_on` returns. Calling
    /// `block_on` again will resume previously spawned tasks.
    ///
    /// # Panics
    ///
    /// This function panics if the provided future panics, or if called within an
    /// asynchronous execution context.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    ///
    /// // Create the runtime
    /// let rt  = Runtime::new().unwrap();
    ///
    /// // Execute the future, blocking the current thread until completion
    /// rt.block_on(async {
    ///     println!("hello");
    /// });
    /// # }
    /// ```
    ///
    /// [handle]: fn@Handle::block_on
    #[track_caller]
    pub fn block_on<F: Future>(&self, future: F) -> F::Output {
        let fut_size = mem::size_of::<F>();
        if fut_size > BOX_FUTURE_THRESHOLD {
            self.block_on_inner(Box::pin(future), SpawnMeta::new_unnamed(fut_size))
        } else {
            self.block_on_inner(future, SpawnMeta::new_unnamed(fut_size))
        }
    }

    #[track_caller]
    fn block_on_inner<F: Future>(&self, future: F, _meta: SpawnMeta<'_>) -> F::Output {
        #[cfg(all(
            tokio_unstable,
            feature = "taskdump",
            feature = "rt",
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
        ))]
        let future = super::task::trace::Trace::root(future);

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        let future = crate::util::trace::task(
            future,
            "block_on",
            _meta,
            crate::runtime::task::Id::next().as_u64(),
        );

        let _enter = self.enter();

        match &self.scheduler {
            Scheduler::CurrentThread(exec) => exec.block_on(&self.handle.inner, future),
            #[cfg(feature = "rt-multi-thread")]
            Scheduler::MultiThread(exec) => exec.block_on(&self.handle.inner, future),
        }
    }

    /// Enters the runtime context.
    ///
    /// This allows you to construct types that must have an executor
    /// available on creation such as [`Sleep`] or [`TcpStream`]. It will
    /// also allow you to call methods such as [`tokio::spawn`].
    ///
    /// [`Sleep`]: struct@crate::time::Sleep
    /// [`TcpStream`]: struct@crate::net::TcpStream
    /// [`tokio::spawn`]: fn@crate::spawn
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    /// use tokio::task::JoinHandle;
    ///
    /// fn function_that_spawns(msg: String) -> JoinHandle<()> {
    ///     // Had we not used `rt.enter` below, this would panic.
    ///     tokio::spawn(async move {
    ///         println!("{}", msg);
    ///     })
    /// }
    ///
    /// fn main() {
    ///     let rt = Runtime::new().unwrap();
    ///
    ///     let s = "Hello World!".to_string();
    ///
    ///     // By entering the context, we tie `tokio::spawn` to this executor.
    ///     let _guard = rt.enter();
    ///     let handle = function_that_spawns(s);
    ///
    ///     // Wait for the task before we end the test.
    ///     rt.block_on(handle).unwrap();
    /// }
    /// # }
    /// ```
    pub fn enter(&self) -> EnterGuard<'_> {
        self.handle.enter()
    }

    /// Shuts down the runtime, waiting for at most `duration` for all spawned
    /// work to stop.
    ///
    /// See the [struct level documentation](Runtime#shutdown) for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    /// use tokio::task;
    ///
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// fn main() {
    ///    let runtime = Runtime::new().unwrap();
    ///
    ///    runtime.block_on(async move {
    ///        task::spawn_blocking(move || {
    ///            thread::sleep(Duration::from_secs(10_000));
    ///        });
    ///    });
    ///
    ///    runtime.shutdown_timeout(Duration::from_millis(100));
    /// }
    /// # }
    /// ```
    pub fn shutdown_timeout(mut self, duration: Duration) {
        // Wakeup and shutdown all the worker threads
        self.handle.inner.shutdown();
        self.blocking_pool.shutdown(Some(duration));
    }

    /// Shuts down the runtime, without waiting for any spawned work to stop.
    ///
    /// This can be useful if you want to drop a runtime from within another runtime.
    /// Normally, dropping a runtime will block indefinitely for spawned blocking tasks
    /// to complete, which would normally not be permitted within an asynchronous context.
    /// By calling `shutdown_background()`, you can drop the runtime from such a context.
    ///
    /// Note however, that because we do not wait for any blocking tasks to complete, this
    /// may result in a resource leak (in that any blocking tasks are still running until they
    /// return.
    ///
    /// See the [struct level documentation](Runtime#shutdown) for more details.
    ///
    /// This function is equivalent to calling `shutdown_timeout(Duration::from_nanos(0))`.
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio::runtime::Runtime;
    ///
    /// fn main() {
    ///    let runtime = Runtime::new().unwrap();
    ///
    ///    runtime.block_on(async move {
    ///        let inner_runtime = Runtime::new().unwrap();
    ///        // ...
    ///        inner_runtime.shutdown_background();
    ///    });
    /// }
    /// # }
    /// ```
    pub fn shutdown_background(self) {
        self.shutdown_timeout(Duration::from_nanos(0));
    }

    /// Returns a view that lets you get information about how the runtime
    /// is performing.
    pub fn metrics(&self) -> crate::runtime::RuntimeMetrics {
        self.handle.metrics()
    }

    /// Resume workers so they poll tasks freely in the background.
    ///
    /// Call [`pause`](Self::pause) to stop them again. By default a
    /// runtime starts with its workers running, exactly like stock
    /// tokio; a runtime built with
    /// [`Builder::start_workers_paused`](crate::runtime::Builder::start_workers_paused)
    /// starts paused instead — its workers only run after `resume()`
    /// or during [`run_until_stalled`](Self::run_until_stalled).
    #[cfg(feature = "rt-multi-thread")]
    pub fn resume(&self) {
        self.handle.inner.resume();
    }

    /// Pause workers so they stop polling tasks.
    ///
    /// Tasks are NOT cancelled — they remain in their current state
    /// and will continue when `resume()` or `run_until_stalled()` is
    /// called.
    ///
    /// **Blocking barrier.** This call blocks — with no timeout —
    /// until every worker has yielded its current task back to the
    /// scheduler (workers check the pause flag between polls, never
    /// mid-poll). It is pump-control machinery: call it only from a
    /// dedicated control thread, never from a worker (a worker cannot
    /// yield while blocked here — deadlock) and never while holding
    /// anything a worker needs to finish its poll. A task that does
    /// not yield cooperatively stalls this barrier indefinitely.
    #[cfg(feature = "rt-multi-thread")]
    pub fn pause(&self) {
        self.handle.inner.pause();
    }

    /// Drive spawned tasks for up to `budget`, from a clean-start
    /// barrier: first waits for every worker to yield (the
    /// [`pause`](Self::pause) barrier — **unbounded**, see its doc),
    /// then resumes them and waits until either the runtime stalls or
    /// the budget expires.
    ///
    /// Returns `DriveOutcome::Stalled` if all workers became idle (all
    /// tasks are either completed or parked on I/O/channels/wakers)
    /// before the budget expired — the caller's quiescence signal: at
    /// that instant no spawned task had ready work. Returns
    /// `DriveOutcome::BudgetExhausted` if the budget expired while
    /// tasks were still active.
    ///
    /// **Workers keep running after this method returns** — on either
    /// outcome. There is no trailing pause: between calls the workers
    /// continue to poll whatever is or becomes ready (draining
    /// channels, handling late wakes), and the clean-start barrier of
    /// the *next* call is what re-establishes a known state. A
    /// `Stalled` outcome is therefore a statement about that moment,
    /// not a frozen state: a task woken afterwards is picked up in the
    /// background. Callers that need the workers actually stopped must
    /// call [`pause`](Self::pause) explicitly.
    ///
    /// This is designed for engines that need step-by-step control
    /// over execution: run a batch of work, inspect results, decide
    /// whether to continue. Build the runtime with
    /// [`Builder::start_workers_paused`](crate::runtime::Builder::start_workers_paused)
    /// so nothing runs before the first pump; on a default-built
    /// runtime the workers free-run between construction and the
    /// first call (this method's opening barrier still establishes a
    /// clean start either way).
    #[cfg(feature = "rt-multi-thread")]
    pub fn run_until_stalled(&self, budget: Duration) -> DriveOutcome {
        use std::time::Instant;

        self.handle.inner.pause();
        self.handle.inner.resume();

        let deadline = Instant::now() + budget;
        let stalled = self.handle.inner.wait_for_stall(deadline);

        // Leave workers running in the background so they can
        // drain channels between run_until_stalled calls; the next
        // call's pause barrier re-establishes a known state.

        if stalled {
            DriveOutcome::Stalled
        } else {
            DriveOutcome::BudgetExhausted
        }
    }

}

/// Outcome of [`Runtime::run_until_stalled`].
#[cfg(feature = "rt-multi-thread")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriveOutcome {
    /// All workers are idle — no task has work to do.
    Stalled,
    /// The time budget expired while tasks were still active.
    BudgetExhausted,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        match &mut self.scheduler {
            Scheduler::CurrentThread(current_thread) => {
                // This ensures that tasks spawned on the current-thread
                // runtime are dropped inside the runtime's context.
                let _guard = context::try_set_current(&self.handle.inner);
                current_thread.shutdown(&self.handle.inner);
            }
            #[cfg(feature = "rt-multi-thread")]
            Scheduler::MultiThread(multi_thread) => {
                // Unpause workers so they can observe the shutdown signal.
                self.handle.inner.resume();
                // The threaded scheduler drops its tasks on its worker threads, which is
                // already in the runtime's context.
                multi_thread.shutdown(&self.handle.inner);
            }
        }
    }
}

impl std::panic::UnwindSafe for Runtime {}

impl std::panic::RefUnwindSafe for Runtime {}

fn display_eq(d: impl std::fmt::Display, s: &str) -> bool {
    use std::fmt::Write;

    struct FormatEq<'r> {
        remainder: &'r str,
        unequal: bool,
    }

    impl<'r> Write for FormatEq<'r> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            if !self.unequal {
                if let Some(new_remainder) = self.remainder.strip_prefix(s) {
                    self.remainder = new_remainder;
                } else {
                    self.unequal = true;
                }
            }
            Ok(())
        }
    }

    let mut fmt_eq = FormatEq {
        remainder: s,
        unequal: false,
    };
    let _ = write!(fmt_eq, "{d}");
    fmt_eq.remainder.is_empty() && !fmt_eq.unequal
}

/// Checks whether the given error was emitted by Tokio when shutting down its runtime.
///
/// # Examples
///
/// ```
/// # #[cfg(not(target_family = "wasm"))]
/// # {
/// use tokio::runtime::Runtime;
/// use tokio::net::TcpListener;
///
/// fn main() {
///     let rt1 = Runtime::new().unwrap();
///     let rt2 = Runtime::new().unwrap();
///
///     let listener = rt1.block_on(async {
///         TcpListener::bind("127.0.0.1:0").await.unwrap()
///     });
///
///     drop(rt1);
///
///     rt2.block_on(async {
///         let res = listener.accept().await;
///         assert!(res.is_err());
///         assert!(tokio::runtime::is_rt_shutdown_err(res.as_ref().unwrap_err()));
///     });
/// }
/// # }
/// ```
pub fn is_rt_shutdown_err(err: &io::Error) -> bool {
    if let Some(inner) = err.get_ref() {
        err.kind() == io::ErrorKind::Other
            && inner.source().is_none()
            && display_eq(inner, RUNTIME_SHUTTING_DOWN_ERROR)
    } else {
        false
    }
}

#[cfg(all(test, feature = "rt-multi-thread"))]
mod run_until_stalled_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn build_rt(workers: usize) -> Runtime {
        Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .unwrap()
    }

    /// A default-built runtime behaves like stock tokio: spawned
    /// tasks run with no `resume()`/`run_until_stalled()` anywhere —
    /// the workers (and with them the I/O and timer drivers) start
    /// unpaused.
    #[test]
    fn default_build_runs_freely() {
        for workers in [1, 2] {
            let rt = build_rt(workers);
            let (tx, rx) = crate::sync::oneshot::channel();
            rt.handle().spawn(async move {
                let _ = tx.send(42u32);
            });
            let got = rt.block_on(async {
                crate::time::timeout(Duration::from_secs(5), rx).await
            });
            assert_eq!(
                got.expect("workers run without a pump").unwrap(),
                42,
                "workers={workers}"
            );
        }
    }

    /// `Builder::start_workers_paused(true)` is the pump-controlled
    /// opt-in: nothing spawned runs until the first release.
    #[test]
    fn paused_start_is_opt_in() {
        let ran = Arc::new(AtomicBool::new(false));
        let ran2 = ran.clone();
        let rt = Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .start_workers_paused(true)
            .build()
            .unwrap();
        rt.handle().spawn(async move {
            ran2.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !ran.load(Ordering::SeqCst),
            "a paused-start runtime runs nothing before its first release"
        );
        let outcome = rt.run_until_stalled(Duration::from_secs(2));
        assert!(ran.load(Ordering::SeqCst));
        assert_eq!(outcome, DriveOutcome::Stalled);
    }

    /// A task woken via `tokio::sync::watch` inside a `tokio::select!`
    /// must be detected as non-idle by `run_until_stalled`. This
    /// reproduced a race where the driver saw workers as idle before
    /// the woken task was picked up.
    #[test]
    fn watch_wakes_select_parked_task() {
        for workers in [1, 2] {
            let (watch_tx, mut watch_rx) = crate::sync::watch::channel(0u32);
            let woke = Arc::new(AtomicBool::new(false));
            let woke2 = woke.clone();

            let rt = build_rt(workers);

            let (mpsc_tx, mut mpsc_rx) = crate::sync::mpsc::unbounded_channel::<()>();
            rt.handle().spawn(async move {
                let _keep = mpsc_tx;
                // A biased two-way select, written out by hand — the
                // `select!` macro cannot be invoked from within the
                // tokio crate itself (its expansion refers to exported
                // macros by absolute path, which rustc denies for
                // macro-expanded `macro_export` macros).
                use std::future::Future;
                use std::task::Poll;
                let mut changed = std::pin::pin!(watch_rx.changed());
                let mut recv = std::pin::pin!(async { mpsc_rx.recv().await });
                std::future::poll_fn(|cx| {
                    if changed.as_mut().poll(cx).is_ready() {
                        woke2.store(true, Ordering::SeqCst);
                        return Poll::Ready(());
                    }
                    recv.as_mut().poll(cx).map(|_| ())
                })
                .await;
            });

            rt.handle().spawn(async move {
                crate::task::yield_now().await;
                watch_tx.send_modify(|v| *v = 1);
            });

            let outcome = rt.run_until_stalled(Duration::from_secs(2));

            assert!(woke.load(Ordering::SeqCst),
                "workers={workers}: watch notification should wake the select-parked task");
            assert_eq!(outcome, DriveOutcome::Stalled);
        }
    }

    /// Workers must survive rapid pump cycling without losing a resume
    /// wake. `resume()` once flipped `paused` and notified without
    /// holding `pause_mutex`; a worker that had observed `paused ==
    /// true` but not yet parked missed the notify and slept until the
    /// *next* resume — and when every worker missed it, a caller
    /// awaiting a background task (an actor reply) between pumps
    /// deadlocked. This hammers the pause/resume handshake against an
    /// actor round-trip; pre-fix it wedged within a few hundred
    /// cycles.
    #[test]
    fn rapid_pump_cycles_never_lose_the_resume_wake() {
        for workers in [1, 2, 4] {
            let rt = build_rt(workers);

            // An actor: replies to every request. Only makes progress
            // while workers are awake.
            let (req_tx, mut req_rx) =
                crate::sync::mpsc::unbounded_channel::<crate::sync::oneshot::Sender<u64>>();
            rt.handle().spawn(async move {
                let mut n = 0u64;
                while let Some(reply) = req_rx.recv().await {
                    n += 1;
                    let _ = reply.send(n);
                }
            });

            for i in 1..=500u64 {
                // Pump with a tiny budget — the worker herd races the
                // pause/resume handshake every cycle.
                rt.run_until_stalled(Duration::from_micros(50));

                // Between pumps, the workers are supposed to keep
                // running: an actor round-trip must complete without
                // another pump. A lost resume wake parks the herd and
                // this poll times out.
                let (tx, mut rx) = crate::sync::oneshot::channel();
                req_tx.send(tx).unwrap();
                let start = std::time::Instant::now();
                let got = loop {
                    match rx.try_recv() {
                        Ok(v) => break Some(v),
                        Err(crate::sync::oneshot::error::TryRecvError::Empty) => {
                            if start.elapsed() > Duration::from_secs(5) {
                                break None;
                            }
                            std::thread::yield_now();
                        }
                        Err(_) => break None,
                    }
                };
                // On failure, dump the pause machinery's state — the
                // wedge signature is `paused=false active=0`: the herd
                // idle-parked with the wake bookkeeping corrupted.
                if got != Some(i) {
                    if let crate::runtime::scheduler::Handle::MultiThread(ref h) =
                        rt.handle().inner
                    {
                        let (paused, active, has_started) = h.pause_state();
                        eprintln!(
                            "pause state: paused={paused} active={active} has_started={has_started}"
                        );
                    }
                }
                assert_eq!(
                    got,
                    Some(i),
                    "workers={workers}: cycle {i} — the actor never replied; \
                     a wake was lost and the worker herd is parked"
                );
            }
        }
    }
}
