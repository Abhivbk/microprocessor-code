use std::sync::{mpsc, Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

mod pythonspawn;

use pythonspawn::runpythonfile_stream;

fn main() {
    println!("Background worker starting...");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        println!("Shutdown signal received.");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let (tx, rx) = mpsc::channel::<(&'static str, String)>();

    // YOLO runs inside python/test.py. Keep the standalone detector disabled
    // to avoid loading and executing the same model twice.
    // let tx_cam = tx.clone();
    // thread::spawn(move || {
    //     runpythonfile_stream(
    //         "python/cone_detection/camera_cone_detection.py",
    //         "camera_cone_detection.py",
    //         tx_cam,
    //     );
    // });



    let tx_test = tx.clone();
    thread::spawn(move || {
        runpythonfile_stream(
            "python/test.py",
            "test.py",
            tx_test,
        );
    });

    drop(tx);

    while running.load(Ordering::SeqCst) {
        if let Ok((tag, line)) = rx.recv_timeout(Duration::from_millis(500)) {
            println!("[{tag}] {line}");
        }
    }

    println!("Background worker shutting down.");
}
