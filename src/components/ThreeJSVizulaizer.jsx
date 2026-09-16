import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'; 
import io from 'socket.io-client';


const ThreeJSVisualizer = ({ isGraphActive , viewMode }) => {

    const mountRef = useRef(null);
    const sceneRef = useRef(null);
    const modelRef = useRef(null);
    const wsRef = useRef(null);
    const animationRef = useRef(null);
    const [isDataReceived, setIsDataReceived] = useState(false);
    const targetPositionRef = useRef({ x: 0, y: 0, z: 0 });
    const targetRotationRef = useRef({ x: 0, y: 0, z: 0 });
    const mouseRef = useRef({ x: 0, y: 0, isDown: false });
    const cameraRef = useRef({ radius: 10, phi: Math.PI / 4, theta: Math.PI / 4 });

    const lerp = (start, end, factor) => {
        return start + (end - start) * factor;
    };

    const updateModelTransform = () => {
        if (!modelRef.current) return;

        const model = modelRef.current;
        const targetPos = targetPositionRef.current;
        const targetRot = targetRotationRef.current;
        const POSITION_SMOOTHING = 0.1;
        const ROTATION_SMOOTHING = 0.05;

        // Update position
        model.position.x = lerp(model.position.x, targetPos.y, POSITION_SMOOTHING);
        model.position.y = lerp(model.position.y, targetPos.x, POSITION_SMOOTHING);
        model.position.z = lerp(model.position.z, targetPos.z, POSITION_SMOOTHING);

        // Update rotation (in radians)
        model.rotation.x = lerp(model.rotation.x, targetRot.y * (Math.PI / 180), ROTATION_SMOOTHING);
        model.rotation.y = lerp(model.rotation.y, targetRot.x * (Math.PI / 180), ROTATION_SMOOTHING);
        model.rotation.z = lerp(model.rotation.z, targetRot.z * (Math.PI / 180), ROTATION_SMOOTHING);
    };

    const updateCameraPosition = () => {
        if (!sceneRef.current) return;
        const { camera } = sceneRef.current;
        const { radius, phi, theta } = cameraRef.current;

        camera.position.x = radius * Math.sin(phi) * Math.cos(theta);
        camera.position.y = radius * Math.cos(phi);
        camera.position.z = radius * Math.sin(phi) * Math.sin(theta);
        camera.lookAt(0, 0, 0);
    };
    const connectSocketIO = () => {
        const socket = io('http://127.0.0.1:9262', {
            transports: ["polling" ],
            withCredentials: true,
            upgrade: false,

        });

        socket.on('connect_error', (error) => {
            console.error("Connection error:", error);
        });

        socket.on('receive_imu_gyro', (data) => {
            try {
                if (data.position || data.kalman) {
                    setIsDataReceived(true);
                }
                if (data.position) {
                    targetPositionRef.current = {
                        x: data.position.x,
                        y: data.position.y,
                        z: data.position.z
                    };
                }
                if (data.kalman) {
                    targetRotationRef.current = {
                        x: data.kalman.x,
                        y: data.kalman.y,
                        z: data.kalman.z
                    };
                }
            } catch (error) {
                console.error("Error processing data:", error);
                console.error("Problematic data:", data);
            }
        });

        socket.on('disconnect_sensor', () => {
            targetPositionRef.current = { x: 0, y: 0, z: 0 };
            targetRotationRef.current = { x: 0, y: 0, z: 0 };
            setIsDataReceived(false);
        });

        wsRef.current = socket;
        return socket;
    };
    useEffect(() => {

        if (isGraphActive) {
            connectSocketIO();
        } else {
            if (wsRef.current) {
                wsRef.current.close();
            }
        }
    }, [isGraphActive]);


    const handleResize = () => {
        const { camera, renderer } = sceneRef.current;
        const width = mountRef.current.offsetWidth;
        const height = mountRef.current.offsetHeight;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    };

    useEffect(() => {
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        // Camera setup
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        updateCameraPosition();

        let width = mountRef.current.offsetWidth
        let height = mountRef.current.offsetHeight

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        renderer.domElement.style.borderRadius = "10px";
        mountRef.current.appendChild(renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 5);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        // Grid
        const gridHelper = new THREE.GridHelper(20, 60, 0x444444, 0x222222);
        gridHelper.position.set(0, 0, 0);
        scene.add(gridHelper);

        // Ground plane
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x222222,
            transparent: true,
            opacity: 0.5
        });


        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(0, 0, 0);  
        ground.receiveShadow = true;
        scene.add(ground);

        const loader = new GLTFLoader();
        const modelPath = '/assets/3d_image.glb';
        loader.load(
            modelPath,
            (gltf) => {
                const model = gltf.scene;
                model.scale.set(2, 2, 2);
                modelRef.current = model;
                model.castShadow = true;
                model.receiveShadow = true;
                scene.add(model);
                const arrowLength = 2; 

                // X axis (Red)
                const xAxisArrowPostive = new THREE.ArrowHelper(
                    new THREE.Vector3(1, 0, 0),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0xff0000
                );
                model.add(xAxisArrowPostive);

                // X axis (Red)
                const xAxisArrowNegative = new THREE.ArrowHelper(
                    new THREE.Vector3(-1, 0, 0),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0xff0000
                );
                model.add(xAxisArrowNegative);

                // Y axis (Green)
                const yAxisArrowPostive = new THREE.ArrowHelper(
                    new THREE.Vector3(0, 1, 0),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0x00ff00
                );
                model.add(yAxisArrowPostive);

                // Y axis (Green)
                const yAxisArrowNegative = new THREE.ArrowHelper(
                    new THREE.Vector3(0, -1, 0),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0x00ff00
                );
                model.add(yAxisArrowNegative);

                // Z axis (Blue)
                const zAxisArrowPostive = new THREE.ArrowHelper(
                    new THREE.Vector3(0, 0, 1),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0x0000ff
                );
                model.add(zAxisArrowPostive);

                // Z axis (Blue)
                const zAxisArrowNegative = new THREE.ArrowHelper(
                    new THREE.Vector3(0, 0, -1),
                    new THREE.Vector3(0, 0, 0),
                    arrowLength,
                    0x0000ff
                );
                model.add(zAxisArrowNegative);
            },
            undefined,
            (error) => {
                console.error('An error occurred while loading the model:', error);
            }
        );

        const axesHelper = new THREE.AxesHelper(8);
        scene.add(axesHelper);
        sceneRef.current = { scene, camera, renderer };
        const handleMouseDown = (e) => {
            mouseRef.current = {
                x: e.clientX,
                y: e.clientY,
                isDown: true
            };
        };

        const handleMouseUp = () => {
            mouseRef.current.isDown = false;
        };

        const handleMouseMove = (e) => {
            if (!mouseRef.current.isDown) return;

            const deltaX = e.clientX - mouseRef.current.x;
            const deltaY = e.clientY - mouseRef.current.y;

            cameraRef.current.theta += deltaX * 0.01;
            cameraRef.current.phi = Math.max(0.1, Math.min(Math.PI - 0.1,
                cameraRef.current.phi + deltaY * 0.01));

            mouseRef.current.x = e.clientX;
            mouseRef.current.y = e.clientY;
        };

        const handleWheel = (e) => {
            e.preventDefault();
            const zoomSpeed = 0.05;
            const zoomFactor = e.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;

            cameraRef.current.radius = Math.max(5, Math.min(20,
                cameraRef.current.radius * zoomFactor
            ));
        };
        // Add line graph labels
        const createLabel = (text, position, color) => {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 128;
            canvas.height = 32;

            context.fillStyle = 'rgba(0, 0, 0, 0.7)';
            context.fillRect(0, 0, canvas.width, canvas.height);

            context.font = '24px Arial';
            context.fillStyle = color;
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(text, canvas.width / 2, canvas.height / 2);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(material);
            sprite.position.copy(position);
            sprite.scale.set(1, 0.25, 1);

            return sprite;
        };

        // Add labels
        scene.add(createLabel('X-Axis', new THREE.Vector3(5, 0.5, -5), '#ff0000'));
        scene.add(createLabel('Y-Axis', new THREE.Vector3(5, 5, 0.5), '#00ff00'));
        scene.add(createLabel('Z-Axis', new THREE.Vector3(-5, 0.5, 5), '#0000ff'));


        // Add event listeners
        renderer.domElement.addEventListener('mousedown', handleMouseDown);
        renderer.domElement.addEventListener('mouseup', handleMouseUp);
        renderer.domElement.addEventListener('mousemove', handleMouseMove);
        renderer.domElement.addEventListener('wheel', handleWheel, { passive: false });

        // Animation loop
        const animate = (time) => {
            if (modelRef.current) updateModelTransform();
            updateCameraPosition();
            renderer.render(scene, camera);
            animationRef.current = requestAnimationFrame(animate);
        };
        animate();

        window.addEventListener('resize', handleResize);

        handleResize();

        const currentMount = mountRef.current;
        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            renderer.domElement.removeEventListener('mousedown', handleMouseDown);
            renderer.domElement.removeEventListener('mouseup', handleMouseUp);
            renderer.domElement.removeEventListener('mousemove', handleMouseMove);
            renderer.domElement.removeEventListener('wheel', handleWheel);

            if (currentMount) {
                currentMount.removeChild(renderer.domElement);
            }
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            renderer.dispose();
        };

    }, []);

    useEffect(() => {
        handleResize()
    }, [viewMode]);

    return (
        <div
            ref={mountRef}
            className='three-js-visualizer'

        />
    );
};

export default ThreeJSVisualizer;