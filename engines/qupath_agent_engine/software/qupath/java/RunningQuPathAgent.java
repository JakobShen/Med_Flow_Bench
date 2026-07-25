package qupath_agent_engine.software.qupath.java;

import java.awt.Shape;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.io.File;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.instrument.Instrumentation;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public final class RunningQuPathAgent {
    private static final double VIEWER_COORD_MIN = 0.0;
    private static final double VIEWER_COORD_MAX = 999.0;

    private RunningQuPathAgent() {}

    public static void agentmain(String args, Instrumentation instrumentation) throws Exception {
        run(args);
    }

    public static void agentmain(String args) throws Exception {
        run(args);
    }

    public static void premain(String args, Instrumentation instrumentation) throws Exception {
        run(args);
    }

    private static void run(String rawArgs) throws Exception {
        String action = value(rawArgs, "action");
        String projectPath = value(rawArgs, "project");
        String imageName = value(rawArgs, "imageName");
        String imagePath = value(rawArgs, "imagePath");
        String imageType = value(rawArgs, "imageType");
        String direction = value(rawArgs, "direction");
        String amount = value(rawArgs, "amount");
        String unit = value(rawArgs, "unit");
        String mode = value(rawArgs, "mode");
        String qpdataPath = value(rawArgs, "qpdataPath");
        String x = value(rawArgs, "x");
        String y = value(rawArgs, "y");
        String outputPath = value(rawArgs, "outputPath");
        String errorPath = value(rawArgs, "errorPath");

        String normalizedAction = action == null || action.isBlank() ? "load_case" : action.trim().toLowerCase();

        try {
            Class<?> platformClass = Class.forName("javafx.application.Platform");
            Method runLater = platformClass.getMethod("runLater", Runnable.class);
            CountDownLatch latch = new CountDownLatch(1);

            runLater.invoke(
                null,
                (Runnable) () -> {
                    try {
                        switch (normalizedAction) {
                            case "query":
                            case "query_state":
                                queryState(outputPath);
                                break;
                            case "open_project":
                                loadCase(projectPath, imageName, null, outputPath, normalizedAction);
                                break;
                            case "load_image":
                                loadCase(null, null, imagePath, outputPath, normalizedAction);
                                break;
                            case "set_image_type":
                                setImageType(imageType, outputPath);
                                break;
                            case "discard_unsaved_changes":
                                discardUnsavedChanges(outputPath);
                                break;
                            case "zoom":
                                zoomViewer(direction, amount, unit, x, y, outputPath);
                                break;
                            case "pan":
                                panViewer(direction, amount, unit, mode, x, y, outputPath);
                                break;
                            case "viewer_point_to_wsi_coordinate":
                                viewerPointToWsiCoordinate(x, y, outputPath);
                                break;
                            case "score_point_against_qpdata":
                                scorePointAgainstQpdata(qpdataPath, x, y, outputPath);
                                break;
                            case "load_case":
                            default:
                                loadCase(projectPath, imageName, imagePath, outputPath, normalizedAction);
                                break;
                        }
                    } catch (Throwable t) {
                        writeText(errorPath, stackTrace(t));
                    } finally {
                        latch.countDown();
                    }
                }
            );

            if (!latch.await(30, TimeUnit.SECONDS)) {
                throw new RuntimeException("Timed out while waiting for the QuPath JavaFX thread.");
            }
        } catch (Throwable t) {
            writeText(errorPath, stackTrace(t));
            throw t;
        }
    }

    private static void loadCase(String projectPath, String imageName, String imagePath, String outputPath, String actionName) throws Exception {
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Class<?> projectIoClass = Class.forName("qupath.lib.projects.ProjectIO");
        Class<?> projectEntryClass = Class.forName("qupath.lib.projects.ProjectImageEntry");
        Object qupath = requireQuPath(guiClass);

        if (projectPath != null && !projectPath.isBlank()) {
            Object project = projectIoClass
                .getMethod("loadProject", File.class, Class.class)
                .invoke(null, new File(projectPath), java.awt.image.BufferedImage.class);
            guiClass.getMethod("setProject", projectClass).invoke(qupath, project);

            if (imageName != null && !imageName.isBlank()) {
                @SuppressWarnings("unchecked")
                List<Object> imageList = (List<Object>) projectClass.getMethod("getImageList").invoke(project);
                Object matchedEntry = null;
                for (Object entry : imageList) {
                    String currentName = String.valueOf(projectEntryClass.getMethod("getImageName").invoke(entry));
                    if (imageName.equals(currentName)) {
                        matchedEntry = entry;
                        break;
                    }
                }
                if (matchedEntry == null) {
                    throw new IllegalArgumentException("No project image entry matched imageName: " + imageName);
                }
                guiClass.getMethod("openImageEntry", projectEntryClass).invoke(qupath, matchedEntry);
            }
        } else if (imagePath != null && !imagePath.isBlank()) {
            guiClass.getMethod("setProject", projectClass).invoke(qupath, new Object[] {null});
            Object viewer = guiClass.getMethod("getViewer").invoke(qupath);
            Class<?> viewerClass = Class.forName("qupath.lib.gui.viewer.QuPathViewer");
            guiClass.getMethod("openImage", viewerClass, String.class).invoke(qupath, viewer, imagePath);
        }

        if (outputPath != null && !outputPath.isBlank()) {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("ok", true);
            payload.put("action", actionName);
            if (projectPath != null && !projectPath.isBlank()) {
                payload.put("requested_project_path", projectPath);
            }
            if (imageName != null && !imageName.isBlank()) {
                payload.put("requested_project_image_name", imageName);
            }
            if (imagePath != null && !imagePath.isBlank()) {
                payload.put("requested_image_path", imagePath);
            }
            payload.put("state", stateMap(qupath, guiClass, projectClass));
            writeText(outputPath, toJson(payload));
        }
    }

    private static void queryState(String outputPath) throws Exception {
        if (outputPath == null || outputPath.isBlank()) {
            return;
        }
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "query_state");
        payload.put("state", stateMap(qupath, guiClass, projectClass));
        writeText(outputPath, toJson(payload));
    }

    private static void setImageType(String imageTypeName, String outputPath) throws Exception {
        if (imageTypeName == null || imageTypeName.isBlank()) {
            throw new IllegalArgumentException("imageType is required");
        }
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        Map<String, Object> beforeState = stateMap(qupath, guiClass, projectClass);
        Class<?> qpClass = Class.forName("qupath.lib.scripting.QP");
        boolean changed = Boolean.TRUE.equals(qpClass.getMethod("setImageType", String.class).invoke(null, imageTypeName));
        Map<String, Object> afterState = stateMap(qupath, guiClass, projectClass);

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", changed);
        payload.put("action", "set_image_type");
        payload.put("requested_image_type", imageTypeName);
        payload.put("supported_image_types", supportedImageTypes());
        payload.put("state_before", beforeState);
        payload.put("state_after", afterState);
        payload.put("state", afterState);
        if (!changed) {
            payload.put("error", "QP.setImageType returned false");
        }
        writeText(outputPath, toJson(payload));
    }

    private static void discardUnsavedChanges(String outputPath) throws Exception {
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        @SuppressWarnings("unchecked")
        List<Object> viewers = (List<Object>) guiClass.getMethod("getAllViewers").invoke(qupath);

        int clearedCount = 0;
        int changedBeforeCount = 0;
        for (Object viewer : viewers) {
            if (viewer == null) {
                continue;
            }
            Object imageData = viewer.getClass().getMethod("getImageData").invoke(viewer);
            if (imageData == null) {
                continue;
            }
            Boolean changed = (Boolean) imageData.getClass().getMethod("isChanged").invoke(imageData);
            if (Boolean.TRUE.equals(changed)) {
                changedBeforeCount++;
                imageData.getClass().getMethod("setChanged", boolean.class).invoke(imageData, false);
                clearedCount++;
            }
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "discard_unsaved_changes");
        payload.put("changed_viewer_count_before", changedBeforeCount);
        payload.put("cleared_viewer_count", clearedCount);
        payload.put("state", stateMap(qupath, guiClass, projectClass));
        writeText(outputPath, toJson(payload));
    }

    private static void zoomViewer(String direction, String amountRaw, String unit, String xRaw, String yRaw, String outputPath) throws Exception {
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        Object imageData = guiClass.getMethod("getImageData").invoke(qupath);
        if (imageData == null) {
            throw new IllegalStateException("No image is open in QuPath.");
        }
        Object viewer = requireViewer(guiClass, qupath);
        Class<?> viewerClass = viewer.getClass();

        String normalizedDirection = normalizeLower(direction);
        String normalizedUnit = normalizeZoomUnit(unit);
        double amount = parsePositiveDouble(amountRaw, "amount");
        if (normalizedUnit == null) {
            throw new IllegalArgumentException("Unsupported zoom unit: " + unit);
        }
        if (!Objects.equals(normalizedUnit, "magnification")
                && !Objects.equals(normalizedDirection, "in")
                && !Objects.equals(normalizedDirection, "out")) {
            throw new IllegalArgumentException("Unsupported zoom direction: " + direction);
        }

        Map<String, Object> beforeState = stateMap(qupath, guiClass, projectClass);
        Map<String, Object> beforeViewer = viewerMapFromState(beforeState);
        double beforeMagnification = mapDouble(beforeViewer, "magnification");
        double beforeDownsample = mapDouble(beforeViewer, "downsample");

        Object server = imageData.getClass().getMethod("getServer").invoke(imageData);
        double imageWidth = server == null ? 0.0 : toDouble(server.getClass().getMethod("getWidth").invoke(server));
        double imageHeight = server == null ? 0.0 : toDouble(server.getClass().getMethod("getHeight").invoke(server));
        Map<String, Object> focusPoint = resolveViewerPointToImagePoint(
            viewer,
            viewerClass,
            imageWidth,
            imageHeight,
            xRaw,
            yRaw,
            true
        );
        double focusImageX = toDouble(focusPoint.get("image_x"));
        double focusImageY = toDouble(focusPoint.get("image_y"));

        double currentDownsample = toDouble(viewerClass.getMethod("getDownsampleFactor").invoke(viewer));
        boolean zoomApplied = false;
        if (Objects.equals(normalizedUnit, "steps")) {
            int steps = Math.max(1, (int)Math.round(amount));
            double defaultZoomFactor = toDouble(invokeNoArgs(viewer, "getDefaultZoomFactor"));
            if (Double.isFinite(currentDownsample) && currentDownsample > 0.0
                    && Double.isFinite(defaultZoomFactor) && defaultZoomFactor > 1.0) {
                double targetDownsample = Objects.equals(normalizedDirection, "in")
                    ? currentDownsample / Math.pow(defaultZoomFactor, steps)
                    : currentDownsample * Math.pow(defaultZoomFactor, steps);
                setViewerDownsample(viewer, viewerClass, targetDownsample, focusImageX, focusImageY);
                zoomApplied = true;
            }
            if (!zoomApplied) {
                if (Objects.equals(normalizedDirection, "in")) {
                    viewerClass.getMethod("zoomIn", int.class).invoke(viewer, steps);
                } else {
                    viewerClass.getMethod("zoomOut", int.class).invoke(viewer, steps);
                }
                zoomApplied = true;
            }
        } else if (Objects.equals(normalizedUnit, "relative_factor")) {
            if (amount <= 1.0) {
                throw new IllegalArgumentException("Zoom factor must be > 1.0");
            }
            double targetDownsample = Objects.equals(normalizedDirection, "in")
                ? currentDownsample / amount
                : currentDownsample * amount;
            setViewerDownsample(viewer, viewerClass, targetDownsample, focusImageX, focusImageY);
            zoomApplied = true;
        } else {
            if (amount <= 0.0) {
                throw new IllegalArgumentException("Target magnification must be > 0");
            }
            double fullMagnification = toDouble(viewerClass.getMethod("getFullMagnification").invoke(viewer));
            if (Double.isFinite(fullMagnification) && fullMagnification > 0.0) {
                double targetDownsample = fullMagnification / amount;
                setViewerDownsample(viewer, viewerClass, targetDownsample, focusImageX, focusImageY);
                zoomApplied = true;
            }
            if (!zoomApplied) {
                try {
                    viewerClass.getMethod("setMagnification", double.class).invoke(viewer, amount);
                    zoomApplied = true;
                } catch (NoSuchMethodException noSuchMethodException) {
                    throw new IllegalStateException("Unable to determine target magnification or target downsample.");
                }
            }
        }

        if (!zoomApplied) {
            throw new IllegalStateException("QuPath did not apply the requested zoom.");
        }

        Shape displayedShape = (Shape) viewerClass.getMethod("getDisplayedRegionShape").invoke(viewer);
        Rectangle2D displayedBounds = displayedShape == null ? null : displayedShape.getBounds2D();
        double viewportWidth = displayedBounds == null ? 0.0 : displayedBounds.getWidth();
        double viewportHeight = displayedBounds == null ? 0.0 : displayedBounds.getHeight();
        double targetCenterX = clampCenter(focusImageX, imageWidth, viewportWidth);
        double targetCenterY = clampCenter(focusImageY, imageHeight, viewportHeight);
        try {
            viewerClass.getMethod("requestStopMoving").invoke(viewer);
        } catch (Exception ignored) {
        }
        viewerClass.getMethod("setCenterPixelLocation", double.class, double.class).invoke(viewer, targetCenterX, targetCenterY);

        Map<String, Object> afterState = stateMap(qupath, guiClass, projectClass);
        Map<String, Object> afterViewer = viewerMapFromState(afterState);
        double afterMagnification = mapDouble(afterViewer, "magnification");
        double afterDownsample = mapDouble(afterViewer, "downsample");

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "zoom");
        payload.put("direction", normalizedDirection);
        payload.put("amount", amount);
        payload.put("unit", normalizedUnit);
        if (Objects.equals(normalizedUnit, "magnification")) {
            payload.put("target_magnification", amount);
        }
        if (Double.isFinite(beforeMagnification)) {
            payload.put("before_magnification", beforeMagnification);
        }
        if (Double.isFinite(afterMagnification)) {
            payload.put("after_magnification", afterMagnification);
        }
        if (Double.isFinite(beforeDownsample)) {
            payload.put("before_downsample", beforeDownsample);
        }
        if (Double.isFinite(afterDownsample)) {
            payload.put("after_downsample", afterDownsample);
        }
        addFocusPointFields(payload, focusPoint);
        payload.put("target_center_x", targetCenterX);
        payload.put("target_center_y", targetCenterY);
        payload.put("effective_direction", effectiveZoomDirection(beforeMagnification, afterMagnification));
        payload.put("state_before", beforeState);
        payload.put("state_after", afterState);
        payload.put("state", afterState);
        writeText(outputPath, toJson(payload));
    }

    private static void panViewer(String direction, String amountRaw, String unit, String mode, String xRaw, String yRaw, String outputPath) throws Exception {
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        Object viewer = requireViewer(guiClass, qupath);
        Class<?> viewerClass = viewer.getClass();
        Object imageData = guiClass.getMethod("getImageData").invoke(qupath);
        if (imageData == null) {
            throw new IllegalStateException("No image is open in QuPath.");
        }
        Class<?> imageDataClass = Class.forName("qupath.lib.images.ImageData");
        Object server = imageDataClass.getMethod("getServer").invoke(imageData);
        if (server == null) {
            throw new IllegalStateException("No image server is available for the current image.");
        }
        Class<?> serverClass = server.getClass();
        double imageWidth = toDouble(serverClass.getMethod("getWidth").invoke(server));
        double imageHeight = toDouble(serverClass.getMethod("getHeight").invoke(server));

        String normalizedMode = normalizePanMode(mode);
        if (normalizedMode == null) {
            normalizedMode = hasText(xRaw) || hasText(yRaw) ? "set_center" : "directional";
        }

        Map<String, Object> beforeState = stateMap(qupath, guiClass, projectClass);
        Shape displayedShape = (Shape) viewerClass.getMethod("getDisplayedRegionShape").invoke(viewer);
        Rectangle2D displayedBounds = displayedShape == null ? null : displayedShape.getBounds2D();
        double viewportWidth = displayedBounds == null ? 0.0 : displayedBounds.getWidth();
        double viewportHeight = displayedBounds == null ? 0.0 : displayedBounds.getHeight();

        double targetX;
        double targetY;
        double dx = Double.NaN;
        double dy = Double.NaN;
        Double amount = null;
        String normalizedDirection = null;
        String normalizedUnit = null;
        Map<String, Object> focusPoint = null;

        if (Objects.equals(normalizedMode, "set_center")) {
            focusPoint = resolveViewerPointToImagePoint(
                viewer,
                viewerClass,
                imageWidth,
                imageHeight,
                xRaw,
                yRaw,
                false
            );
            double focusImageX = toDouble(focusPoint.get("image_x"));
            double focusImageY = toDouble(focusPoint.get("image_y"));
            targetX = clampCenter(focusImageX, imageWidth, viewportWidth);
            targetY = clampCenter(focusImageY, imageHeight, viewportHeight);
        } else if (Objects.equals(normalizedMode, "directional")) {
            normalizedDirection = normalizeLower(direction);
            normalizedUnit = normalizeLower(unit);
            amount = parsePositiveDouble(amountRaw, "amount");
            if (!Objects.equals(normalizedUnit, "viewport_fraction") && !Objects.equals(normalizedUnit, "image_pixels")) {
                throw new IllegalArgumentException("Unsupported pan unit: " + unit);
            }
            double[] directionVector = directionVector(normalizedDirection);
            dx = directionVector[0];
            dy = directionVector[1];
            if (Objects.equals(normalizedUnit, "viewport_fraction")) {
                if (displayedBounds == null) {
                    throw new IllegalStateException("Displayed viewer region is unavailable.");
                }
                dx *= amount * displayedBounds.getWidth();
                dy *= amount * displayedBounds.getHeight();
            } else {
                dx *= amount;
                dy *= amount;
            }
            double centerX = toDouble(viewerClass.getMethod("getCenterPixelX").invoke(viewer));
            double centerY = toDouble(viewerClass.getMethod("getCenterPixelY").invoke(viewer));
            targetX = clampCenter(centerX + dx, imageWidth, viewportWidth);
            targetY = clampCenter(centerY + dy, imageHeight, viewportHeight);
        } else {
            throw new IllegalArgumentException("Unsupported pan mode: " + mode);
        }

        try {
            viewerClass.getMethod("requestStopMoving").invoke(viewer);
        } catch (Exception ignored) {
        }
        viewerClass.getMethod("setCenterPixelLocation", double.class, double.class).invoke(viewer, targetX, targetY);
        Map<String, Object> afterState = stateMap(qupath, guiClass, projectClass);

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "pan");
        payload.put("mode", normalizedMode);
        if (normalizedDirection != null) {
            payload.put("direction", normalizedDirection);
        }
        if (amount != null) {
            payload.put("amount", amount);
        }
        if (normalizedUnit != null) {
            payload.put("unit", normalizedUnit);
        }
        if (Double.isFinite(dx)) {
            payload.put("delta_x", dx);
        }
        if (Double.isFinite(dy)) {
            payload.put("delta_y", dy);
        }
        if (focusPoint != null) {
            addFocusPointFields(payload, focusPoint);
        }
        payload.put("target_center_x", targetX);
        payload.put("target_center_y", targetY);
        payload.put("state_before", beforeState);
        payload.put("state_after", afterState);
        payload.put("state", afterState);
        writeText(outputPath, toJson(payload));
    }

    private static void viewerPointToWsiCoordinate(String xRaw, String yRaw, String outputPath) throws Exception {
        Class<?> guiClass = Class.forName("qupath.lib.gui.QuPathGUI");
        Class<?> projectClass = Class.forName("qupath.lib.projects.Project");
        Object qupath = requireQuPath(guiClass);
        Object viewer = requireViewer(guiClass, qupath);
        Class<?> viewerClass = viewer.getClass();

        Object imageData = guiClass.getMethod("getImageData").invoke(qupath);
        if (imageData == null) {
            throw new IllegalStateException("No image is open in QuPath.");
        }
        Object server = imageData.getClass().getMethod("getServer").invoke(imageData);
        if (server == null) {
            throw new IllegalStateException("No image server is available for the current image.");
        }
        Class<?> serverClass = server.getClass();
        double imageWidth = toDouble(serverClass.getMethod("getWidth").invoke(server));
        double imageHeight = toDouble(serverClass.getMethod("getHeight").invoke(server));
        Map<String, Object> resolvedPoint = resolveViewerPointToImagePoint(
            viewer,
            viewerClass,
            imageWidth,
            imageHeight,
            xRaw,
            yRaw,
            false
        );

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "viewer_point_to_wsi_coordinate");
        payload.putAll(resolvedPoint);
        payload.put("coordinate_system", "qupath_full_resolution_image_pixels");
        payload.put("image_width", imageWidth);
        payload.put("image_height", imageHeight);
        payload.put("state", stateMap(qupath, guiClass, projectClass));
        writeText(outputPath, toJson(payload));
    }

    private static void scorePointAgainstQpdata(String qpdataPath, String xRaw, String yRaw, String outputPath) throws Exception {
        if (qpdataPath == null || qpdataPath.isBlank()) {
            throw new IllegalArgumentException("qpdataPath is required");
        }
        double x = parseFiniteDouble(xRaw, "x");
        double y = parseFiniteDouble(yRaw, "y");
        Path path = Path.of(qpdataPath);
        Object hierarchy = readHierarchyFromQpdata(path);
        List<Map<String, Object>> hits = new ArrayList<>();
        int nAnnotationsTotal = 0;
        int nAnnotationsWithClassification = 0;
        int nAnnotationsWithNormalizedClassification = 0;
        Map<String, Object> normalizedHistogram = new LinkedHashMap<>();

        Iterable<?> annotations = annotationObjectsFromHierarchy(hierarchy);
        if (annotations != null) {
            for (Object annotation : annotations) {
                if (annotation == null) {
                    continue;
                }
                nAnnotationsTotal++;
                String classificationName = annotationClassificationName(annotation);
                if (!classificationName.isBlank()) {
                    nAnnotationsWithClassification++;
                    String classificationNormalized = normalizeClassification(classificationName);
                    if (!classificationNormalized.isBlank()) {
                        nAnnotationsWithNormalizedClassification++;
                        incrementHistogramCount(normalizedHistogram, classificationNormalized);
                    }
                }
                Object roi = invokeNoArgs(annotation, "getROI");
                if (roi == null) {
                    continue;
                }
                if (!roiContainsPoint(roi, x, y)) {
                    continue;
                }
                hits.add(annotationHitMap(annotation, roi));
            }
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ok", true);
        payload.put("action", "score_point_against_qpdata");
        payload.put("qpdata_path", qpdataPath);
        payload.put("x", x);
        payload.put("y", y);
        payload.put("hit_any", !hits.isEmpty());
        payload.put("n_hits", hits.size());
        payload.put("hits", hits);
        payload.put(
            "annotation_summary",
            qpdataAnnotationSummary(
                nAnnotationsTotal,
                nAnnotationsWithClassification,
                nAnnotationsWithNormalizedClassification,
                normalizedHistogram
            )
        );
        writeText(outputPath, toJson(payload));
    }

    private static Object readHierarchyFromQpdata(Path qpdataPath) throws Exception {
        Class<?> pathIoClass = Class.forName("qupath.lib.io.PathIO");
        Method method = preferredReadHierarchyMethod(pathIoClass);
        if (method == null) {
            throw new NoSuchMethodException("PathIO.readHierarchy was not found");
        }
        Class<?> paramType = method.getParameterTypes()[0];
        Object argument;
        if (paramType.isAssignableFrom(Path.class)) {
            argument = qpdataPath;
        } else if (paramType.isAssignableFrom(File.class)) {
            argument = qpdataPath.toFile();
        } else if (paramType.isAssignableFrom(String.class)) {
            argument = qpdataPath.toString();
        } else if (paramType.isAssignableFrom(InputStream.class)) {
            try (InputStream input = Files.newInputStream(qpdataPath)) {
                Object hierarchy = method.invoke(null, input);
                if (hierarchy == null) {
                    throw new IllegalStateException("PathIO.readHierarchy returned null for " + qpdataPath);
                }
                return hierarchy;
            }
        } else {
            throw new IllegalStateException("Unsupported PathIO.readHierarchy parameter type: " + paramType.getName());
        }
        Object hierarchy = method.invoke(null, argument);
        if (hierarchy == null) {
            throw new IllegalStateException("PathIO.readHierarchy returned null for " + qpdataPath);
        }
        return hierarchy;
    }

    private static Method preferredReadHierarchyMethod(Class<?> pathIoClass) {
        Method best = null;
        int bestRank = Integer.MAX_VALUE;
        for (Method candidate : pathIoClass.getMethods()) {
            if (!Objects.equals(candidate.getName(), "readHierarchy")) {
                continue;
            }
            if (candidate.getParameterCount() != 1) {
                continue;
            }
            int rank = hierarchyReadMethodRank(candidate.getParameterTypes()[0]);
            if (rank < bestRank) {
                best = candidate;
                bestRank = rank;
            }
        }
        return best;
    }

    private static int hierarchyReadMethodRank(Class<?> paramType) {
        if (paramType == null) {
            return Integer.MAX_VALUE;
        }
        if (paramType.isAssignableFrom(Path.class)) {
            return 0;
        }
        if (paramType.isAssignableFrom(File.class)) {
            return 1;
        }
        if (paramType.isAssignableFrom(String.class)) {
            return 2;
        }
        if (paramType.isAssignableFrom(InputStream.class)) {
            return 3;
        }
        return Integer.MAX_VALUE;
    }

    private static Iterable<?> annotationObjectsFromHierarchy(Object hierarchy) {
        Object annotations = invokeNoArgs(hierarchy, "getAnnotationObjects");
        Iterable<?> iterable = iterableOf(annotations);
        if (iterable != null) {
            return iterable;
        }
        Object flattened = invokeNoArgs(hierarchy, "getFlattenedObjectList");
        if (!(flattened instanceof Iterable<?>)) {
            return null;
        }
        List<Object> annotationsOnly = new ArrayList<>();
        for (Object item : (Iterable<?>)flattened) {
            if (item == null) {
                continue;
            }
            Object isAnnotation = invokeNoArgs(item, "isAnnotation");
            if (isAnnotation != null) {
                if (toBoolean(isAnnotation)) {
                    annotationsOnly.add(item);
                }
                continue;
            }
            Object roi = invokeNoArgs(item, "getROI");
            if (roi != null) {
                annotationsOnly.add(item);
            }
        }
        return annotationsOnly;
    }

    private static boolean roiContainsPoint(Object roi, double x, double y) {
        try {
            Method contains = roi.getClass().getMethod("contains", double.class, double.class);
            Object value = contains.invoke(roi, x, y);
            if (value != null) {
                return toBoolean(value);
            }
        } catch (Exception ignored) {
        }
        try {
            Method contains = roi.getClass().getMethod("contains", int.class, int.class);
            Object value = contains.invoke(roi, (int)Math.round(x), (int)Math.round(y));
            if (value != null) {
                return toBoolean(value);
            }
        } catch (Exception ignored) {
        }
        Object shape = invokeNoArgs(roi, "getShape");
        if (shape instanceof Shape) {
            return ((Shape)shape).contains(x, y);
        }
        return false;
    }

    private static Map<String, Object> annotationHitMap(Object annotation, Object roi) {
        Map<String, Object> hit = new LinkedHashMap<>();
        hit.put("object_type", annotation.getClass().getSimpleName());
        hit.put("name", stringValue(invokeNoArgs(annotation, "getName")));
        Object id = invokeNoArgs(annotation, "getID");
        if (id != null) {
            hit.put("object_id", String.valueOf(id));
        }
        String classificationName = annotationClassificationName(annotation);
        hit.put("classification", classificationName);
        hit.put("classification_normalized", normalizeClassification(classificationName));
        hit.put("roi_type", stringValue(invokeNoArgs(roi, "getRoiName")));
        hit.put("centroid_x", toDouble(invokeNoArgs(roi, "getCentroidX")));
        hit.put("centroid_y", toDouble(invokeNoArgs(roi, "getCentroidY")));
        return hit;
    }

    private static String annotationClassificationName(Object annotation) {
        Object pathClass = invokeNoArgs(annotation, "getPathClass");
        String classificationName = stringValue(invokeNoArgs(pathClass, "getName"));
        if (classificationName.isBlank()) {
            classificationName = stringValue(pathClass);
        }
        return classificationName;
    }

    private static Map<String, Object> qpdataAnnotationSummary(
        int nAnnotationsTotal,
        int nAnnotationsWithClassification,
        int nAnnotationsWithNormalizedClassification,
        Map<String, Object> normalizedHistogram
    ) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("n_annotations_total", nAnnotationsTotal);
        summary.put("n_annotations_with_classification", nAnnotationsWithClassification);
        summary.put("n_annotations_with_normalized_classification", nAnnotationsWithNormalizedClassification);
        summary.put("classification_histogram_normalized", normalizedHistogram);
        return summary;
    }

    private static void incrementHistogramCount(Map<String, Object> histogram, String key) {
        Object current = histogram.get(key);
        int count = current instanceof Number ? ((Number) current).intValue() : 0;
        histogram.put(key, count + 1);
    }

    private static Object requireQuPath(Class<?> guiClass) throws Exception {
        Object qupath = guiClass.getMethod("getInstance").invoke(null);
        if (qupath == null) {
            throw new IllegalStateException("QuPathGUI.getInstance() returned null.");
        }
        return qupath;
    }

    private static Object requireViewer(Class<?> guiClass, Object qupath) throws Exception {
        Object viewer = guiClass.getMethod("getViewer").invoke(qupath);
        if (viewer == null) {
            throw new IllegalStateException("QuPath viewer is unavailable.");
        }
        return viewer;
    }

    private static Map<String, Object> stateMap(Object qupath, Class<?> guiClass, Class<?> projectClass) throws Exception {
        Map<String, Object> state = new LinkedHashMap<>();
        Object project = guiClass.getMethod("getProject").invoke(qupath);
        Object imageData = guiClass.getMethod("getImageData").invoke(qupath);

        state.put("project_name", project == null ? "" : String.valueOf(projectClass.getMethod("getName").invoke(project)));
        state.put("project_path", project == null ? "" : invokeToString(projectClass, project, "getPath"));
        state.put("gui", guiStateMap(guiClass, qupath));

        if (imageData == null) {
            state.put("displayed_image_name", "");
            state.put("server_path", "");
            state.put("image_type_value", "");
            state.put("image_type_label", "");
            state.put("viewer", new LinkedHashMap<>());
            return state;
        }

        Class<?> imageDataClass = Class.forName("qupath.lib.images.ImageData");
        Object imageType = imageDataClass.getMethod("getImageType").invoke(imageData);
        state.put("displayed_image_name", String.valueOf(guiClass.getMethod("getDisplayedImageName", imageDataClass).invoke(qupath, imageData)));
        Object server = imageDataClass.getMethod("getServer").invoke(imageData);
        if (server != null) {
            state.put("server_path", invokeToString(server.getClass(), server, "getPath"));
        } else {
            state.put("server_path", "");
        }
        if (imageType instanceof Enum<?>) {
            state.put("image_type_value", ((Enum<?>) imageType).name());
            state.put("image_type_label", String.valueOf(imageType));
        } else if (imageType != null) {
            state.put("image_type_value", String.valueOf(imageType));
            state.put("image_type_label", String.valueOf(imageType));
        } else {
            state.put("image_type_value", "");
            state.put("image_type_label", "");
        }

        state.put("viewer", viewerStateMap(guiClass, qupath, server));
        return state;
    }

    private static Map<String, Object> viewerStateMap(Class<?> guiClass, Object qupath, Object server) throws Exception {
        Map<String, Object> viewerState = new LinkedHashMap<>();
        Object viewer = guiClass.getMethod("getViewer").invoke(qupath);
        if (viewer == null) {
            return viewerState;
        }
        Class<?> viewerClass = viewer.getClass();
        viewerState.put("center_pixel_x", toDouble(viewerClass.getMethod("getCenterPixelX").invoke(viewer)));
        viewerState.put("center_pixel_y", toDouble(viewerClass.getMethod("getCenterPixelY").invoke(viewer)));
        viewerState.put("downsample", toDouble(viewerClass.getMethod("getDownsampleFactor").invoke(viewer)));
        viewerState.put("magnification", toDouble(viewerClass.getMethod("getMagnification").invoke(viewer)));
        viewerState.put("full_magnification", toDouble(viewerClass.getMethod("getFullMagnification").invoke(viewer)));
        Object zoomToFit = invokeNoArgs(viewer, "getZoomToFit");
        if (zoomToFit != null) {
            viewerState.put("zoom_to_fit", toBoolean(zoomToFit));
        }

        Shape displayedShape = (Shape) viewerClass.getMethod("getDisplayedRegionShape").invoke(viewer);
        if (displayedShape != null) {
            Rectangle2D bounds = displayedShape.getBounds2D();
            Map<String, Object> displayedRegion = new LinkedHashMap<>();
            displayedRegion.put("x", bounds.getX());
            displayedRegion.put("y", bounds.getY());
            displayedRegion.put("width", bounds.getWidth());
            displayedRegion.put("height", bounds.getHeight());
            viewerState.put("displayed_region", displayedRegion);
        } else {
            viewerState.put("displayed_region", new LinkedHashMap<>());
        }

        if (server != null) {
            Class<?> serverClass = server.getClass();
            viewerState.put("image_width", toDouble(serverClass.getMethod("getWidth").invoke(server)));
            viewerState.put("image_height", toDouble(serverClass.getMethod("getHeight").invoke(server)));
        }
        return viewerState;
    }

    private static Map<String, Object> guiStateMap(Class<?> guiClass, Object qupath) {
        Map<String, Object> guiState = new LinkedHashMap<>();
        guiState.put("app_version", stringValue(invokeStaticNoArgs(guiClass, "getVersion")));

        Object stage = invokeNoArgs(qupath, "getStage");
        if (stage != null) {
            guiState.put("window_title", stringValue(invokeNoArgs(stage, "getTitle")));
            putBooleanIfPresent(guiState, "window_showing", invokeNoArgs(stage, "isShowing"));
            putBooleanIfPresent(guiState, "window_focused", invokeNoArgs(stage, "isFocused"));
            putBooleanIfPresent(guiState, "window_iconified", invokeNoArgs(stage, "isIconified"));
            putBooleanIfPresent(guiState, "window_maximized", invokeNoArgs(stage, "isMaximized"));
        }

        putBooleanIfPresent(guiState, "read_only", invokeNoArgs(qupath, "isReadOnly"));
        putBooleanIfPresent(guiState, "ui_blocked", propertyValue(invokeNoArgs(qupath, "uiBlocked")));
        putBooleanIfPresent(guiState, "analysis_pane_visible", propertyValue(invokeNoArgs(qupath, "showAnalysisPaneProperty")));
        putBooleanIfPresent(guiState, "input_display_visible", propertyValue(invokeNoArgs(qupath, "showInputDisplayProperty")));

        Object tabPane = invokeNoArgs(qupath, "getAnalysisTabPane");
        if (tabPane != null) {
            List<Map<String, Object>> analysisTabs = new ArrayList<>();
            Object selectionModel = invokeNoArgs(tabPane, "getSelectionModel");
            Object selectedTab = selectionModel == null ? null : invokeNoArgs(selectionModel, "getSelectedItem");
            String selectedTabTitle = labelFromControl(selectedTab);
            Iterable<?> tabs = iterableOf(invokeNoArgs(tabPane, "getTabs"));
            if (tabs != null) {
                for (Object tab : tabs) {
                    String title = labelFromControl(tab);
                    if (title.isBlank()) {
                        continue;
                    }
                    Map<String, Object> tabInfo = new LinkedHashMap<>();
                    tabInfo.put("title", title);
                    tabInfo.put("selected", selectedTab != null && selectedTab.equals(tab));
                    analysisTabs.add(tabInfo);
                }
            }
            guiState.put("analysis_tabs", analysisTabs);
            guiState.put("selected_analysis_tab", selectedTabTitle);
        }

        Object toolManager = invokeNoArgs(qupath, "getToolManager");
        if (toolManager != null) {
            guiState.put("selected_tool_name", pathToolName(invokeNoArgs(toolManager, "getSelectedTool")));
            guiState.put("previous_tool_name", pathToolName(invokeNoArgs(toolManager, "getPreviousSelectedTool")));
            putBooleanIfPresent(guiState, "tool_switching_enabled", invokeNoArgs(toolManager, "isToolSwitchingEnabled"));
            putBooleanIfPresent(guiState, "selected_tool_locked", propertyValue(invokeNoArgs(toolManager, "lockSelectedToolProperty")));

            List<Map<String, Object>> availableTools = new ArrayList<>();
            Iterable<?> tools = iterableOf(invokeNoArgs(toolManager, "getTools"));
            if (tools != null) {
                for (Object tool : tools) {
                    String toolName = pathToolName(tool);
                    if (toolName.isBlank()) {
                        continue;
                    }
                    Map<String, Object> toolInfo = new LinkedHashMap<>();
                    toolInfo.put("name", toolName);
                    availableTools.add(toolInfo);
                }
            }
            guiState.put("available_tools", availableTools);
        }

        Object toolBar = invokeNoArgs(qupath, "getToolBar");
        if (toolBar != null) {
            List<Map<String, Object>> toolbarItems = new ArrayList<>();
            Iterable<?> items = iterableOf(invokeNoArgs(toolBar, "getItems"));
            if (items != null) {
                for (Object item : items) {
                    if (item == null) {
                        continue;
                    }
                    Object isVisible = invokeNoArgs(item, "isVisible");
                    if (isVisible != null && !toBoolean(isVisible)) {
                        continue;
                    }
                    String label = toolbarItemLabel(item);
                    if (label.isBlank()) {
                        continue;
                    }
                    Map<String, Object> itemInfo = new LinkedHashMap<>();
                    itemInfo.put("label", label);
                    itemInfo.put("type", item.getClass().getSimpleName());
                    Object selected = invokeNoArgs(item, "isSelected");
                    if (selected != null) {
                        itemInfo.put("selected", toBoolean(selected));
                    }
                    Object disabled = invokeNoArgs(item, "isDisable");
                    if (disabled != null) {
                        itemInfo.put("disabled", toBoolean(disabled));
                    }
                    toolbarItems.add(itemInfo);
                }
            }
            guiState.put("toolbar_items", toolbarItems);
        }

        return guiState;
    }

    private static Map<String, Object> viewerMapFromState(Map<String, Object> state) {
        Object viewer = state.get("viewer");
        if (viewer instanceof Map<?, ?>) {
            @SuppressWarnings("unchecked")
            Map<String, Object> viewerMap = (Map<String, Object>)viewer;
            return viewerMap;
        }
        return new LinkedHashMap<>();
    }

    private static String normalizeZoomUnit(String unit) {
        String normalized = normalizeLower(unit);
        switch (normalized) {
            case "magnification":
            case "target_magnification":
            case "absolute":
                return "magnification";
            case "relative_factor":
            case "factor":
            case "relative":
            case "multiplier":
                return "relative_factor";
            case "steps":
                return "steps";
            default:
                return null;
        }
    }


    private static String normalizePanMode(String mode) {
        String normalized = normalizeLower(mode);
        switch (normalized) {
            case "direction":
            case "directional":
            case "relative":
            case "move":
                return "directional";
            case "set_center":
            case "center":
            case "recenter":
            case "target_center":
            case "viewer_point":
                return "set_center";
            default:
                return null;
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }

    private static void setViewerDownsample(
        Object viewer,
        Class<?> viewerClass,
        double targetDownsample,
        double focusImageX,
        double focusImageY
    ) throws Exception {
        if (!Double.isFinite(targetDownsample) || targetDownsample <= 0.0) {
            throw new IllegalArgumentException("targetDownsample must be > 0");
        }
        Method methodWithFocus = findMethod(viewerClass, "setDownsampleFactor", 3);
        if (methodWithFocus != null && Double.isFinite(focusImageX) && Double.isFinite(focusImageY)) {
            methodWithFocus.invoke(viewer, targetDownsample, focusImageX, focusImageY);
            return;
        }
        viewerClass.getMethod("setDownsampleFactor", double.class).invoke(viewer, targetDownsample);
    }

    private static Map<String, Object> resolveViewerPointToImagePoint(
        Object viewer,
        Class<?> viewerClass,
        double imageWidth,
        double imageHeight,
        String xRaw,
        String yRaw,
        boolean allowBlankToCurrentCenter
    ) throws Exception {
        Map<String, Object> payload = new LinkedHashMap<>();
        if (!hasText(xRaw) && !hasText(yRaw)) {
            if (!allowBlankToCurrentCenter) {
                throw new IllegalArgumentException("x and y are required");
            }
            double centerX = toDouble(viewerClass.getMethod("getCenterPixelX").invoke(viewer));
            double centerY = toDouble(viewerClass.getMethod("getCenterPixelY").invoke(viewer));
            payload.put("coordinate_mode", "current_viewer_center");
            payload.put("image_x", centerX);
            payload.put("image_y", centerY);
            payload.put("inside_image_bounds", pointWithinImageBounds(centerX, centerY, imageWidth, imageHeight));
            payload.put("was_clipped_to_image_bounds", false);
            return payload;
        }
        if (!hasText(xRaw) || !hasText(yRaw)) {
            throw new IllegalArgumentException("x and y must either both be provided or both be omitted");
        }

        double viewerX = parseViewerCoordinate(xRaw, "x");
        double viewerY = parseViewerCoordinate(yRaw, "y");
        Object view = invokeNoArgs(viewer, "getView");
        double viewerWidth = toDouble(view == null ? invokeNoArgs(viewer, "getWidth") : invokeNoArgs(view, "getWidth"));
        double viewerHeight = toDouble(view == null ? invokeNoArgs(viewer, "getHeight") : invokeNoArgs(view, "getHeight"));
        if (!Double.isFinite(viewerWidth) || viewerWidth <= 0.0 || !Double.isFinite(viewerHeight) || viewerHeight <= 0.0) {
            throw new IllegalStateException("The viewer dimensions are unavailable.");
        }

        double componentX = viewerCoordinateToComponentCoordinate(viewerX, viewerWidth);
        double componentY = viewerCoordinateToComponentCoordinate(viewerY, viewerHeight);
        Method converter = findMethod(viewerClass, "componentPointToImagePoint", 4);
        if (converter == null) {
            throw new NoSuchMethodException("QuPathViewer.componentPointToImagePoint was not found");
        }
        Object rawPoint = converter.invoke(viewer, componentX, componentY, null, false);
        Object clippedPoint = converter.invoke(viewer, componentX, componentY, null, true);
        Object outputPoint = clippedPoint != null ? clippedPoint : rawPoint;
        if (outputPoint == null) {
            throw new IllegalStateException("QuPath did not return an image-space coordinate for the requested viewer point.");
        }

        double imageX = toDouble(invokeNoArgs(outputPoint, "getX"));
        double imageY = toDouble(invokeNoArgs(outputPoint, "getY"));
        double rawImageX = rawPoint == null ? Double.NaN : toDouble(invokeNoArgs(rawPoint, "getX"));
        double rawImageY = rawPoint == null ? Double.NaN : toDouble(invokeNoArgs(rawPoint, "getY"));
        double boundsCheckX = Double.isFinite(rawImageX) ? rawImageX : imageX;
        double boundsCheckY = Double.isFinite(rawImageY) ? rawImageY : imageY;
        boolean insideImageBounds = pointWithinImageBounds(boundsCheckX, boundsCheckY, imageWidth, imageHeight);
        boolean wasClippedToImageBounds = false;
        if (rawPoint != null && clippedPoint != null) {
            wasClippedToImageBounds = Math.abs(rawImageX - imageX) > 1e-6 || Math.abs(rawImageY - imageY) > 1e-6;
        }

        payload.put("coordinate_mode", "viewer_point");
        payload.put("viewer_x", viewerX);
        payload.put("viewer_y", viewerY);
        payload.put("component_x", componentX);
        payload.put("component_y", componentY);
        payload.put("viewer_width", viewerWidth);
        payload.put("viewer_height", viewerHeight);
        payload.put("image_x", imageX);
        payload.put("image_y", imageY);
        if (Double.isFinite(rawImageX)) {
            payload.put("raw_image_x", rawImageX);
        }
        if (Double.isFinite(rawImageY)) {
            payload.put("raw_image_y", rawImageY);
        }
        payload.put("inside_image_bounds", insideImageBounds);
        payload.put("was_clipped_to_image_bounds", wasClippedToImageBounds);
        return payload;
    }

    private static void addFocusPointFields(Map<String, Object> payload, Map<String, Object> focusPoint) {
        copyIfPresent(payload, "focus_mode", focusPoint, "coordinate_mode");
        copyIfPresent(payload, "requested_viewer_x", focusPoint, "viewer_x");
        copyIfPresent(payload, "requested_viewer_y", focusPoint, "viewer_y");
        copyIfPresent(payload, "requested_component_x", focusPoint, "component_x");
        copyIfPresent(payload, "requested_component_y", focusPoint, "component_y");
        copyIfPresent(payload, "requested_viewer_width", focusPoint, "viewer_width");
        copyIfPresent(payload, "requested_viewer_height", focusPoint, "viewer_height");
        copyIfPresent(payload, "focus_image_x", focusPoint, "image_x");
        copyIfPresent(payload, "focus_image_y", focusPoint, "image_y");
        copyIfPresent(payload, "focus_raw_image_x", focusPoint, "raw_image_x");
        copyIfPresent(payload, "focus_raw_image_y", focusPoint, "raw_image_y");
        copyIfPresent(payload, "focus_inside_image_bounds", focusPoint, "inside_image_bounds");
        copyIfPresent(payload, "focus_was_clipped_to_image_bounds", focusPoint, "was_clipped_to_image_bounds");
    }

    private static void copyIfPresent(
        Map<String, Object> target,
        String targetKey,
        Map<String, Object> source,
        String sourceKey
    ) {
        if (target == null || source == null || targetKey == null || sourceKey == null) {
            return;
        }
        if (source.containsKey(sourceKey)) {
            target.put(targetKey, source.get(sourceKey));
        }
    }

    private static double parseViewerCoordinate(String raw, String name) {
        double value = parseFiniteDouble(raw, name);
        if (value < VIEWER_COORD_MIN || value > VIEWER_COORD_MAX) {
            throw new IllegalArgumentException(name + " must be between 0 and 999");
        }
        return value;
    }

    private static double viewerCoordinateToComponentCoordinate(double viewerCoordinate, double viewerExtent) {
        if (!Double.isFinite(viewerExtent) || viewerExtent <= 1.0) {
            return 0.0;
        }
        return viewerCoordinate * (viewerExtent - 1.0) / VIEWER_COORD_MAX;
    }

    private static double mapDouble(Map<String, Object> map, String key) {
        if (map == null) {
            return Double.NaN;
        }
        Object value = map.get(key);
        return toDouble(value);
    }

    private static String effectiveZoomDirection(double beforeMagnification, double afterMagnification) {
        if (!Double.isFinite(beforeMagnification) || !Double.isFinite(afterMagnification)) {
            return "unknown";
        }
        if (Math.abs(afterMagnification - beforeMagnification) < 1e-9) {
            return "none";
        }
        return afterMagnification > beforeMagnification ? "in" : "out";
    }

    private static Object invokeStaticNoArgs(Class<?> cls, String methodName) {
        try {
            return cls.getMethod(methodName).invoke(null);
        } catch (Exception e) {
            return null;
        }
    }

    private static Object invokeNoArgs(Object target, String methodName) {
        if (target == null) {
            return null;
        }
        try {
            return target.getClass().getMethod(methodName).invoke(target);
        } catch (Exception e) {
            return null;
        }
    }

    private static Method findMethod(Class<?> cls, String methodName, int parameterCount) {
        if (cls == null) {
            return null;
        }
        for (Method method : cls.getMethods()) {
            if (!Objects.equals(method.getName(), methodName)) {
                continue;
            }
            if (method.getParameterCount() != parameterCount) {
                continue;
            }
            return method;
        }
        return null;
    }

    private static Object propertyValue(Object property) {
        if (property == null) {
            return null;
        }
        Object value = invokeNoArgs(property, "getValue");
        if (value != null) {
            return value;
        }
        return invokeNoArgs(property, "get");
    }

    private static boolean toBoolean(Object value) {
        if (value instanceof Boolean) {
            return (Boolean)value;
        }
        return Boolean.parseBoolean(String.valueOf(value));
    }

    private static void putBooleanIfPresent(Map<String, Object> target, String key, Object rawValue) {
        if (rawValue != null) {
            target.put(key, toBoolean(rawValue));
        }
    }

    private static Iterable<?> iterableOf(Object value) {
        if (value instanceof Iterable<?>) {
            return (Iterable<?>)value;
        }
        return null;
    }

    private static String pathToolName(Object tool) {
        return stringValue(invokeNoArgs(tool, "getName"));
    }

    private static String labelFromControl(Object control) {
        if (control == null) {
            return "";
        }
        String text = stringValue(invokeNoArgs(control, "getText"));
        if (!text.isBlank()) {
            return text;
        }
        text = stringValue(invokeNoArgs(control, "getTitle"));
        if (!text.isBlank()) {
            return text;
        }
        text = stringValue(invokeNoArgs(control, "getAccessibleText"));
        if (!text.isBlank()) {
            return text;
        }
        return "";
    }

    private static String toolbarItemLabel(Object item) {
        if (item == null) {
            return "";
        }
        String label = labelFromControl(item);
        if (!label.isBlank()) {
            return label;
        }
        Object tooltip = invokeNoArgs(item, "getTooltip");
        label = stringValue(invokeNoArgs(tooltip, "getText"));
        if (!label.isBlank()) {
            return label;
        }
        label = stringValue(invokeNoArgs(item, "getId"));
        if (!label.isBlank()) {
            return label;
        }
        String simpleName = item.getClass().getSimpleName();
        if (simpleName == null || simpleName.isBlank()) {
            return "";
        }
        if (simpleName.contains("Separator") || simpleName.equals("Pane") || simpleName.equals("Region")) {
            return "";
        }
        return simpleName;
    }

    private static String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private static Map<String, Object> supportedImageTypes() {
        Map<String, Object> types = new LinkedHashMap<>();
        types.put("BRIGHTFIELD_H_DAB", "Brightfield (H-DAB)");
        types.put("BRIGHTFIELD_H_E", "Brightfield (H&E)");
        types.put("BRIGHTFIELD_OTHER", "Brightfield (other)");
        types.put("FLUORESCENCE", "Fluorescence");
        types.put("OTHER", "Other");
        types.put("UNSET", "Not set");
        return types;
    }

    private static String invokeToString(Class<?> cls, Object target, String methodName) {
        try {
            Object value = cls.getMethod(methodName).invoke(target);
            return value == null ? "" : String.valueOf(value);
        } catch (Exception e) {
            return "";
        }
    }

    private static String normalizeLower(String value) {
        return value == null ? "" : value.trim().toLowerCase();
    }

    private static String normalizeClassification(String raw) {
        String normalized = normalizeClassificationText(raw);
        if (normalized.isBlank()) {
            return "";
        }
        String collapsed = normalized.replace("-", "");
        switch (collapsed) {
            case "n":
                return "N";
            case "pb":
                return "PB";
            case "udh":
                return "UDH";
            case "fea":
                return "FEA";
            case "adh":
                return "ADH";
            case "dcis":
                return "DCIS";
            case "ic":
                return "IC";
            case "benign":
                return "N";
            case "pathologicalbenign":
                return "PB";
            default:
                break;
        }
        String tokenized = normalized.replace('-', ' ').replace('/', ' ');
        if (tokenized.contains(" dcis") || tokenized.startsWith("dcis ") || tokenized.equals("dcis")) {
            return "DCIS";
        }
        if (tokenized.contains(" adh") || tokenized.startsWith("adh ") || tokenized.equals("adh")) {
            return "ADH";
        }
        if (tokenized.contains(" fea") || tokenized.startsWith("fea ") || tokenized.equals("fea")) {
            return "FEA";
        }
        if (tokenized.contains(" udh") || tokenized.startsWith("udh ") || tokenized.equals("udh")) {
            return "UDH";
        }
        if (tokenized.contains(" pb") || tokenized.startsWith("pb ") || tokenized.equals("pb")) {
            return "PB";
        }
        if (tokenized.contains(" ic") || tokenized.startsWith("ic ") || tokenized.equals("ic")) {
            return "IC";
        }
        if (tokenized.contains(" n") || tokenized.startsWith("n ") || tokenized.equals("n")) {
            return "N";
        }
        return "";
    }

    private static String normalizeClassificationText(String raw) {
        String normalized = normalizeLower(raw).replaceAll("[^a-z0-9]+", "-");
        normalized = trimDashes(normalized);
        while (normalized.endsWith("-sure")) {
            normalized = trimDashes(normalized.substring(0, normalized.length() - "-sure".length()));
        }
        return normalized;
    }

    private static String trimDashes(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        int start = 0;
        int end = value.length();
        while (start < end && value.charAt(start) == '-') {
            start++;
        }
        while (end > start && value.charAt(end - 1) == '-') {
            end--;
        }
        return value.substring(start, end);
    }

    private static double parseFiniteDouble(String raw, String name) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException(name + " is required");
        }
        double value;
        try {
            value = Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be numeric: " + raw);
        }
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be finite: " + raw);
        }
        return value;
    }

    private static double parsePositiveDouble(String raw, String name) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException(name + " is required");
        }
        double value = Double.parseDouble(raw);
        if (!(value > 0.0)) {
            throw new IllegalArgumentException(name + " must be > 0");
        }
        return value;
    }

    private static double clampCenter(double center, double imageExtent, double viewportExtent) {
        if (imageExtent <= 0.0) {
            return 0.0;
        }
        if (viewportExtent >= imageExtent) {
            return imageExtent / 2.0;
        }
        double halfViewport = viewportExtent / 2.0;
        double minCenter = Math.max(0.0, halfViewport);
        double maxCenter = Math.max(minCenter, imageExtent - halfViewport);
        return clamp(center, minCenter, maxCenter);
    }

    private static boolean pointWithinImageBounds(double x, double y, double imageWidth, double imageHeight) {
        if (!Double.isFinite(x) || !Double.isFinite(y)) {
            return false;
        }
        if (!(imageWidth > 0.0) || !(imageHeight > 0.0)) {
            return false;
        }
        return x >= 0.0 && y >= 0.0 && x <= imageWidth - 1.0 && y <= imageHeight - 1.0;
    }

    private static double[] directionVector(String direction) {
        switch (direction) {
            case "left":
                return new double[] {-1.0, 0.0};
            case "right":
                return new double[] {1.0, 0.0};
            case "up":
                return new double[] {0.0, -1.0};
            case "down":
                return new double[] {0.0, 1.0};
            case "up_left":
                return new double[] {-1.0, -1.0};
            case "up_right":
                return new double[] {1.0, -1.0};
            case "down_left":
                return new double[] {-1.0, 1.0};
            case "down_right":
                return new double[] {1.0, 1.0};
            default:
                throw new IllegalArgumentException("Unsupported pan direction: " + direction);
        }
    }

    private static double clamp(double value, double min, double max) {
        if (Double.isNaN(value)) {
            return min;
        }
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }

    private static double toDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value == null) {
            return Double.NaN;
        }
        return Double.parseDouble(String.valueOf(value));
    }

    private static String value(String rawArgs, String key) {
        if (rawArgs == null || rawArgs.isBlank()) {
            return null;
        }
        for (String pair : rawArgs.split("&")) {
            if (pair.isBlank()) {
                continue;
            }
            String[] parts = pair.split("=", 2);
            if (parts.length == 2 && parts[0].equals(key)) {
                return java.net.URLDecoder.decode(parts[1], StandardCharsets.UTF_8);
            }
        }
        return null;
    }

    private static void writeText(String outputPath, String text) {
        if (outputPath == null || outputPath.isBlank()) {
            return;
        }
        try {
            Files.writeString(Path.of(outputPath), text == null ? "" : text, StandardCharsets.UTF_8);
        } catch (Exception ignored) {
        }
    }

    private static String stackTrace(Throwable t) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        t.printStackTrace(pw);
        pw.flush();
        return sw.toString();
    }

    private static String toJson(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String) {
            return "\"" + jsonEscape((String) value) + "\"";
        }
        if (value instanceof Double) {
            double d = (Double) value;
            return Double.isFinite(d) ? String.valueOf(d) : "null";
        }
        if (value instanceof Float) {
            float f = (Float) value;
            return Float.isFinite(f) ? String.valueOf(f) : "null";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?>) {
            StringBuilder sb = new StringBuilder();
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                if (!first) {
                    sb.append(',');
                }
                first = false;
                sb.append(toJson(String.valueOf(entry.getKey())));
                sb.append(':');
                sb.append(toJson(entry.getValue()));
            }
            sb.append('}');
            return sb.toString();
        }
        if (value instanceof Iterable<?>) {
            StringBuilder sb = new StringBuilder();
            sb.append('[');
            boolean first = true;
            for (Object item : (Iterable<?>) value) {
                if (!first) {
                    sb.append(',');
                }
                first = false;
                sb.append(toJson(item));
            }
            sb.append(']');
            return sb.toString();
        }
        if (value.getClass().isArray()) {
            List<Object> items = new ArrayList<>();
            int length = java.lang.reflect.Array.getLength(value);
            for (int i = 0; i < length; i++) {
                items.add(java.lang.reflect.Array.get(value, i));
            }
            return toJson(items);
        }
        return toJson(String.valueOf(value));
    }

    private static String jsonEscape(String text) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '\b':
                    sb.append("\\b");
                    break;
                case '\f':
                    sb.append("\\f");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                    break;
            }
        }
        return sb.toString();
    }
}
