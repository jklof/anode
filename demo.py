import time
from core import Graph, Engine
from nodes import SineOscillator, AudioOutput, MonoToStereo


def main():
    print("--- Explicit Stereo Routing Demo ---")

    graph = Graph()

    # 1. Create Nodes
    osc = SineOscillator("Osc1")
    converter = MonoToStereo("Converter")
    speakers = AudioOutput("Speakers")

    # 2. Configure
    osc.params["freq"].set(220.0)

    # 3. Add to Graph
    graph.add_node(osc)
    graph.add_node(converter)
    graph.add_node(speakers)

    # 4. Connect: Osc -> Converter -> Speakers
    print("Connecting: Osc[signal] -> Converter[in]")
    graph.connect(osc, "signal", converter, "in")

    print("Connecting: Converter[out] -> Speakers[audio_in]")
    graph.connect(converter, "out", speakers, "audio_in")

    # 5. Setup Engine
    graph.set_master_clock(speakers)
    engine = Engine(graph)

    # 6. Run
    engine.start()
    print("Running... (220Hz)")

    try:
        time.sleep(2)
        print("Modulating to 440Hz...")
        osc.params["freq"].set(440.0)
        time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        engine.stop()


if __name__ == "__main__":
    main()
