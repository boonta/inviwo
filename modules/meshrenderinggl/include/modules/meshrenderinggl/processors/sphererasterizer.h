/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2021-2022 Inviwo Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************************/

#pragma once

#include <modules/meshrenderinggl/meshrenderingglmoduledefine.h>  // for IVW_MODULE_MESHRENDERI...

#include <inviwo/core/datastructures/light/lightingstate.h>        // for LightingState
#include <inviwo/core/interaction/cameratrackball.h>               // for CameraTrackball
#include <inviwo/core/ports/meshport.h>                            // for MeshFlatMultiInport
#include <inviwo/core/processors/processor.h>                      // for Processor
#include <inviwo/core/processors/processorinfo.h>                  // for ProcessorInfo
#include <inviwo/core/properties/boolproperty.h>                   // for BoolProperty
#include <inviwo/core/properties/cameraproperty.h>                 // for CameraProperty
#include <inviwo/core/properties/compositeproperty.h>              // for CompositeProperty
#include <inviwo/core/properties/optionproperty.h>                 // for OptionProperty
#include <inviwo/core/properties/ordinalproperty.h>                // for FloatProperty, FloatVe...
#include <inviwo/core/properties/simplelightingproperty.h>         // for SimpleLightingProperty
#include <inviwo/core/properties/transferfunctionproperty.h>       // for TransferFunctionProperty
#include <inviwo/core/util/document.h>                             // for Document
#include <inviwo/core/util/glmmat.h>                               // for mat4
#include <inviwo/core/util/glmvec.h>                               // for ivec2, vec4
#include <inviwo/core/util/staticstring.h>                         // for operator+
#include <modules/base/properties/transformlistproperty.h>         // for TransformListProperty
#include <modules/meshrenderinggl/datastructures/rasterization.h>  // for Rasterization
#include <modules/meshrenderinggl/ports/rasterizationport.h>       // for RasterizationOutport

#include <functional>   // for __base, function
#include <memory>       // for shared_ptr
#include <string>       // for operator==, operator+
#include <string_view>  // for operator==
#include <vector>       // for operator!=, vector

namespace inviwo {
class Layer;
class Mesh;
class MeshShaderCache;
class Shader;

/** \docpage{org.inviwo.SphereRasterizer, Sphere Rasterizer}
 * ![](org.inviwo.SphereRasterizer.png?classIdentifier=org.inviwo.SphereRasterizer)
 * Create a rasterization object to render one or more point meshes using spherical glyphs in
 * OpenGL. The glyphs are resolution independent and consist only of a single point.
 *
 * ### Inports
 *   * __geometry__ Input meshes
 *       The input mesh uses the following buffers:
 *          PositionAttrib vec3
 *          ColorAttrib    vec4   (optional will fall-back to use __Custom Color__)
 *          RadiiAttrib    float  (optional will fall-back to use __Custom Radius__)
 *
 * ### Outports
 *   * __rasterization__ rasterization functor rendering  the molecule either opaquely or into
 *                    fragment buffer
 *
 * ### Properties
 *   * __Render Mode__               render only input meshes marked as points or everything
 *   * __Clip Mode__                 defines the handling of spheres clipped at the camera
 *   * __Clip Surface Adjustment__   brighten/darken glyph color on clip surface
 *   * __Shade Clipped Area__        enable illumination computations for the clipped surface
 *   * __Force Radius__              enable a fixed user-defined radius for all spheres
 *   * __Default Radius__            radius of the rendered spheres (in world coordinates)
 *   * __Force Color__               if enabled, all spheres will share the same custom color
 *   * __Default Color__             custom color when overwriting the input colors
 */
class IVW_MODULE_MESHRENDERINGGL_API SphereRasterizer : public Processor {
    friend class SphereRasterization;

public:
    SphereRasterizer();
    virtual ~SphereRasterizer() = default;

    virtual void process() override;

    virtual void initializeResources() override;

    virtual const ProcessorInfo getProcessorInfo() const override;
    static const ProcessorInfo processorInfo_;

private:
    void configureShader(Shader& shader);
    void configureOITShader(Shader& shader);

    enum class RenderMode {
        EntireMesh,  //!< render all vertices of the input mesh as glyphs
        PointsOnly,  //!< render only parts of mesh with DrawType::Points
    };
    /**
     * \enum GlyphClippingMode
     * defines how glyphs are rendering if the first intersection, i.e. the front side,
     * lies behind the near clip plane of the camera.
     */
    enum class GlyphClippingMode {
        Discard,  //!< glyph is not rendered
        Cut,      //!< the cut surface is visible
    };

    MeshFlatMultiInport inport_;
    RasterizationOutport outport_;

    OptionProperty<RenderMode> renderMode_;

    BoolProperty forceOpaque_;
    BoolProperty useUniformAlpha_;
    FloatProperty uniformAlpha_;

    CompositeProperty clipping_;
    OptionProperty<GlyphClippingMode> clipMode_;
    FloatProperty clipShadingFactor_;  //!< multiplied with glyph color for clip surfaces
    BoolProperty shadeClippedArea_;

    CompositeProperty sphereProperties_;
    BoolProperty forceRadius_;
    FloatProperty defaultRadius_;
    BoolProperty forceColor_;
    FloatVec4Property defaultColor_;
    BoolProperty useMetaColor_;
    TransferFunctionProperty metaColor_;

    CameraProperty camera_;
    CameraTrackball trackball_;
    SimpleLightingProperty lighting_;

    TransformListProperty transformSetting_;
    std::shared_ptr<MeshShaderCache> shaders_;
    const bool oitExtensionsAvailable_;
};

/**
 * \brief Functor object that will render molecular data into a fragment list.
 */
class IVW_MODULE_MESHRENDERINGGL_API SphereRasterization : public Rasterization {
public:
    SphereRasterization(const SphereRasterizer& processor);
    virtual void rasterize(const ivec2& imageSize, const mat4& worldMatrixTransform,
                           std::function<void(Shader&)> setUniforms) const override;
    virtual bool usesFragmentLists() const override;
    virtual Document getInfo() const override;
    virtual Rasterization* clone() const override;

protected:
    SphereRasterizer::RenderMode renderMode_;
    float uniformAlpha_;
    const bool forceOpaque_;
    float clipShadingFactor_;

    float defaultRadius_;
    vec4 defaultColor_;
    const Layer* metaColorTF_;
    LightingState lighting_;

    std::shared_ptr<MeshShaderCache> shaders_;
    std::vector<std::shared_ptr<const Mesh>> meshes_;
    bool oitExtensionsAvailable_;
};

}  // namespace inviwo
