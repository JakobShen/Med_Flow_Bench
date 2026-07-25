package qupath_agent_engine.software.qupath.java;

import com.sun.tools.attach.VirtualMachine;
import java.nio.file.Path;
import java.security.ProtectionDomain;

public final class RunningQuPathAttacher {
    private RunningQuPathAttacher() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            throw new IllegalArgumentException("Usage: RunningQuPathAttacher <pid> [encodedArgs]");
        }

        String pid = args[0];
        String encodedArgs = args.length >= 2 ? args[1] : "";
        String jarPath = currentJarPath();

        VirtualMachine vm = VirtualMachine.attach(pid);
        try {
            vm.loadAgent(jarPath, encodedArgs);
        } finally {
            vm.detach();
        }
    }

    private static String currentJarPath() throws Exception {
        ProtectionDomain domain = RunningQuPathAttacher.class.getProtectionDomain();
        return Path.of(domain.getCodeSource().getLocation().toURI()).toString();
    }
}
