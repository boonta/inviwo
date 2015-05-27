/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2014-2015 Inviwo Foundation
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

#ifndef IVW_SPOT_LIGHT_SOURCE_PROCESSOR_H
#define IVW_SPOT_LIGHT_SOURCE_PROCESSOR_H

#include <modules/base/basemoduledefine.h>
#include <inviwo/core/common/inviwo.h>
#include <inviwo/core/datastructures/light/baselightsource.h>
#include <inviwo/core/properties/compositeproperty.h>
#include <inviwo/core/ports/dataoutport.h>
#include <inviwo/core/properties/ordinalproperty.h>
#include <inviwo/core/processors/processor.h>

namespace inviwo {

class SpotLight;

/** \docpage{org.inviwo.Spotlightsource, Spot light source}
 * ![](org.inviwo.Spotlightsource.png?classIdentifier=org.inviwo.Spotlightsource)
 *
 * ...
 * 
 * 
 * 
 * ### Properties
 *   * __Light power (%)__ ...
 *   * __Light size__ ...
 *   * __Light Parameters__ ...
 *   * __Light Cone Radius Angle__ ...
 *   * __Color__ ...
 *   * __Light Source Position__ ...
 *   * __Light Fall Off Angle__ ...
 *
 */
class IVW_MODULE_BASE_API SpotLightSourceProcessor : public Processor {
public:
    SpotLightSourceProcessor();
    virtual ~SpotLightSourceProcessor();

    InviwoProcessorInfo();

protected:
    virtual void process();

    /**
     * Update light source parameters. Transformation will be given in texture space.
     *
     * @param lightSource
     * @return
     */
    void updateSpotLightSource(SpotLight* lightSource);

private:
    DataOutport<LightSource> outport_;

    CompositeProperty lighting_;
    FloatProperty lightPowerProp_;
    FloatVec2Property lightSize_;
    FloatVec4Property lightDiffuse_;
    FloatVec3Property lightPosition_;
    FloatProperty lightConeRadiusAngle_;
    FloatProperty lightFallOffAngle_;

    SpotLight* lightSource_;
};

} // namespace

#endif // IVW_SPOT_LIGHT_SOURCE_PROCESSOR_H
